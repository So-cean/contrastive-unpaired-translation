import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)), persistent=False)

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            # 使用 .clone().detach() 创建独立副本，避免梯度版本冲突
            return F.conv2d(self.pad(inp), self.filt.clone().detach(), stride=self.stride, groups=inp.shape[1])

class Upsample(nn.Module):
    def __init__(self, channels, stride=2, mode='bilinear', **kwargs):
        super().__init__()
        self.scale_factor = stride
        self.mode = mode
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, 
                           mode=self.mode, align_corners=False)  

class LayerNorm(nn.Module):
    """
    支持两种数据格式的 LayerNorm
    channels_last: (B, H, W, C)
    channels_first: (B, C, H, W)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x: [N, H, W, C]
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # [N, 1, 1, C]
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x

class DropPath(nn.Module):
    """Stochastic Depth"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
    
class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 Block"""
    def __init__(self, dim, drop_path=0., kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                               padding=kernel_size//2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = input + self.drop_path(x)
        return x

class ConvNeXtV2Generator(nn.Module):
    """
    ConvNeXt V2 生成器 - 完整修正版
    基于 CUT/CycleGAN 架构，使用 ConvNeXt V2 Block 替代 ResNet Block
    
    特点：
    1. 使用官方正确的 GRN 实现，增强通道间特征竞争
    2. 使用抗锯齿下采样（blur pooling）保持平移等变性
    3. 使用双线性插值上采样，避免棋盘效应
    4. 支持随机深度（Drop Path）正则化
    5. 完全兼容 CUT/CycleGAN 的 NCHW 格式和多层特征提取接口
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None,
                 use_dropout=False, n_blocks=6, padding_type='reflect',
                 no_antialias=False, no_antialias_up=False, opt=None,
                 convnext_kernel_size=7, drop_path_rate=0.1):
        
        super(ConvNeXtV2Generator, self).__init__()
        self.opt = opt
        self.n_blocks = n_blocks
        
        # 构建模型
        model = []
        
        # --- 输入层：7x7 卷积，保持分辨率 ---
        model += [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 LayerNorm(ngf, eps=1e-6, data_format="channels_first"),
                 nn.GELU()]

        n_downsampling = 2
        
        # --- 下采样层：2 次，每次分辨率减半，通道数翻倍 ---
        for i in range(n_downsampling):
            mult = 2 ** i
            in_ch = ngf * mult
            out_ch = ngf * mult * 2
            
            if no_antialias:
                # 普通下采样（简单但可能有混叠）
                model += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                         LayerNorm(in_ch, eps=1e-6, data_format="channels_first"),
                         nn.GELU()]
            else:
                # 抗锯齿下采样：先卷积后模糊降采样（推荐，更平滑）
                model += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                         LayerNorm(out_ch, eps=1e-6, data_format="channels_first"),
                         nn.GELU(),
                         Downsample(out_ch)]

        # --- ConvNeXt V2 Blocks：核心特征提取 ---
        mult = 2 ** n_downsampling
        # 线性增加 drop path 率（从 0 到 drop_path_rate）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        
        for i in range(n_blocks):
            model += [ConvNeXtV2Block(ngf * mult,
                                    drop_path=dpr[i],
                                    kernel_size=convnext_kernel_size)]

        # --- 上采样层：2 次，每次分辨率翻倍，通道数减半 ---
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            in_ch = ngf * mult
            out_ch = int(ngf * mult / 2)
            
            if no_antialias_up:
                # 转置卷积上采样（可能有棋盘效应）
                model += [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2,
                                            padding=1, output_padding=1, bias=False),
                         LayerNorm(out_ch, eps=1e-6, data_format="channels_first"),
                         nn.GELU()]
            else:
                # 插值上采样 + 卷积（推荐，更平滑）
                model += [Upsample(in_ch, stride=2, mode='bilinear'),
                         nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                         LayerNorm(out_ch, eps=1e-6, data_format="channels_first"),
                         nn.GELU()]

        # --- 输出层：7x7 卷积，生成最终图像 ---
        model += [nn.ReflectionPad2d(3),
                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                 nn.Tanh()]  # 输出范围 [-1, 1]

        self.model = nn.Sequential(*model)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 打印模型信息
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[ConvNeXtV2Generator] n_blocks={n_blocks}, "
              f"kernel={convnext_kernel_size}, drop_path={drop_path_rate:.3f}, "
              f"params={n_params:.2f}M, antialias_down={not no_antialias}, "
              f"antialias_up={not no_antialias_up}")

    def _init_weights(self, m):
        """
        ConvNeXt V2 风格的权重初始化
        使用截断正态分布，标准差 0.02
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        # GRN 的 gamma 和 beta 已经在 __init__ 中初始化为 0，不需要额外处理

    def forward(self, input, layers=[], encode_only=False):
        """
        前向传播，完全兼容 CUT 的多层特征提取接口
        
        Args:
            input: 输入图像 [B, C, H, W] (NCHW 格式，CycleGAN 标准)
            layers: 需要提取特征的层索引列表
            encode_only: 是否只返回特征（用于 PatchNCE 损失）
        
        Returns:
            如果 layers 为空：返回生成图像 [B, C, H, W]
            如果 encode_only=True：返回特征列表 [feat1, feat2, ...]
            否则：返回 (生成图像, 特征列表)
        """
        if -1 in layers:
            layers.append(len(self.model))
        
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return feat, feats
        else:
            return self.model(input)
        
# ============= 测试代码 =============
if __name__ == "__main__":
    print("=" * 60)
    print("ConvNeXt V2 Generator 测试")
    print("=" * 60)
    
    # 创建模型
    model = ConvNeXtV2Generator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        n_blocks=9,
        no_antialias=False,
        no_antialias_up=False,
        drop_path_rate=0.1
    )
    
    # 测试输入 (CycleGAN 标准格式)
    x = torch.randn(2, 3, 256, 256)  # [Batch, Channels, Height, Width]
    print(f"\n输入形状: {x.shape}")
    
    # 测试前向传播
    with torch.no_grad():
        # 1. 基本生成
        output = model(x)
        print(f"输出形状: {output.shape}")
        assert output.shape == x.shape, "输���形状应与输入相同"
        
        # 2. 多层特征提取（CUT 模式）
        output, feats = model(x, layers=[4, 8, 12, 16])
        print(f"\n多层特征提取:")
        print(f"  输出形状: {output.shape}")
        for i, feat in enumerate(feats):
            print(f"  特征 {i} 形状: {feat.shape}")
        
        # 3. 仅编码（encode_only 模式）
        feats_only = model(x, layers=[4, 8, 12], encode_only=True)
        print(f"\n仅编码模式:")
        for i, feat in enumerate(feats_only):
            print(f"  特征 {i} 形状: {feat.shape}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！模型与 CycleGAN/CUT 完全兼容")
    print("=" * 60)
    
    # 测试 GRN 单独工作
    print("\n" + "=" * 60)
    print("GRN 单元测试")
    print("=" * 60)
    grn = GRN(dim=256)
    x_nhwc = torch.randn(4, 32, 32, 256)  # [N, H, W, C]
    with torch.no_grad():
        out = grn(x_nhwc)
    print(f"GRN 输入: {x_nhwc.shape}")
    print(f"GRN 输出: {out.shape}")
    assert out.shape == x_nhwc.shape
    print("✅ GRN 测试通过")