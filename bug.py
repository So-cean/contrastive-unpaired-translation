import functools
import torch
import torch.nn as nn
from models.networks import Upsample, Downsample
# ckpt_path = "/home_data/home/songhy2024/contrastive-unpaired-translation/checkpoints/CUT_monai_K2E2_npu_convnext/latest_net_G.pth"
# state_dict = torch.load(ckpt_path, map_location='cpu')
# print(state_dict.keys())


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
                         LayerNorm(out_ch, eps=1e-6, data_format="channels_first"),
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
        
        
    def print_layer_info(self, input_shape=(1, 3, 256, 256), nce_layers=None):
        """
        打印网络层信息，验证 nce_layers 对应的具体位置
        Args:
            input_shape: 测试输入尺寸 (B, C, H, W)
            nce_layers: 传入的 --nce_layers 列表，如 [1, 8, 12, 23, 35]
        """
        print("\n" + "="*80)
        print(f"ConvNeXtGenerator Layer Mapping (n_blocks={self.n_blocks})")
        print("="*80)
        print(f"{'Index':<6} {'Module':<30} {'Output Ch':<12} {'Output Shape':<20} {'Stage'}")
        print("-"*80)
        
        # 注册 hook 捕获每一层输出尺寸
        shapes = {}
        handles = []
        
        def hook_fn(idx):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    shapes[idx] = list(output.shape)
            return hook
        
        for idx, layer in enumerate(self.model):
            handles.append(layer.register_forward_hook(hook_fn(idx)))
        
        # 推理一次获取形状
        device = next(self.parameters()).device
        x = torch.randn(*input_shape).to(device)
        with torch.no_grad():
            _ = self.model(x)
        
        # 移除 hooks
        for h in handles:
            h.remove()
        
        # 打印每一层
        current_stage = "Entry"
        for idx, layer in enumerate(self.model):
            module_name = layer.__class__.__name__
            
            # 判断阶段
            if idx <= 3: current_stage = "Entry"
            elif idx <= 11: current_stage = "Downsample"
            elif idx <= 11 + self.n_blocks: current_stage = f"Bottleneck ({idx-11}/{self.n_blocks})"
            else: current_stage = "Upsample/Exit"
            
            # 获取通道数
            out_ch = shapes.get(idx, ['?'])[1] if idx in shapes else '?'
            shape_str = str(shapes.get(idx, '?'))
            
            # 标记是否为 nce_layers
            marker = ""
            if nce_layers is not None and idx in nce_layers:
                marker = " <<< NCE_LAYER"
            
            print(f"{idx:<6} {module_name:<30} {str(out_ch):<12} {shape_str:<20} {current_stage}{marker}")
        
        print("="*80)
        
        # 特别打印 nce_layers 对应的详细信息
        if nce_layers is not None:
            print(f"\nDetailed NCE Layers (indices: {nce_layers}):")
            for idx in nce_layers:
                if idx < len(self.model):
                    layer = self.model[idx]
                    shape = shapes.get(idx, 'unknown')
                    print(f"  Layer {idx}: {layer.__class__.__name__} -> Output: {shape}")
                    # 如果是卷积层，打印更多细节
                    if hasattr(layer, 'weight') and layer.weight is not None:
                        print(f"            Weight shape: {layer.weight.shape}")
                else:
                    print(f"  WARNING: Layer {idx} out of range (total {len(self.model)} layers)")
            print("="*80)
    
opt = type('', (), {})()  # 创建一个空对象模拟 opt
opt.input_nc = 1
opt.crop_size = 256
opt.nce_layers = '1,8,14,23,30,35'       
netG = ConvNeXtV2Generator(input_nc=opt.input_nc, output_nc=1, ngf=94, n_blocks=24,
                        no_antialias=False, no_antialias_up=False)


# 打印完整层信息（自动推断形状）
nce_layers = [int(i) for i in opt.nce_layers.split(',')]
netG.print_layer_info(input_shape=(1, opt.input_nc, opt.crop_size, opt.crop_size), 
                      nce_layers=nce_layers)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(inplace=False)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(inplace=False)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(inplace=False),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(inplace=False)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(inplace=False)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake
        
    def print_layer_info(self, input_shape=(1, 3, 256, 256), nce_layers=None):
        """
        打印网络层信息，验证 nce_layers 对应的具体位置
        Args:
            input_shape: 测试输入尺寸 (B, C, H, W)
            nce_layers: 传入的 --nce_layers 列表，如 [1, 8, 12, 23, 35]
        """
        print("\n" + "="*80)
        print("="*80)
        print(f"{'Index':<6} {'Module':<30} {'Output Ch':<12} {'Output Shape':<20} {'Stage'}")
        print("-"*80)
        
        # 注册 hook 捕获每一层输出尺寸
        shapes = {}
        handles = []
        
        def hook_fn(idx):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    shapes[idx] = list(output.shape)
            return hook
        
        for idx, layer in enumerate(self.model):
            handles.append(layer.register_forward_hook(hook_fn(idx)))
        
        # 推理一次获取形状
        device = next(self.parameters()).device
        x = torch.randn(*input_shape).to(device)
        with torch.no_grad():
            _ = self.model(x)
        
        # 移除 hooks
        for h in handles:
            h.remove()
        
        # 打印每一层
        current_stage = "Entry"
        for idx, layer in enumerate(self.model):
            module_name = layer.__class__.__name__
            
            
            # 获取通道数
            out_ch = shapes.get(idx, ['?'])[1] if idx in shapes else '?'
            shape_str = str(shapes.get(idx, '?'))
            
            # 标记是否为 nce_layers
            marker = ""
            if nce_layers is not None and idx in nce_layers:
                marker = " <<< NCE LAYER"
            print(f"{idx:<6} {module_name:<30} {str(out_ch):<12} {shape_str:<20} {current_stage}{marker}")
        
        print("="*80)
        
        # 特别打印 nce_layers 对应的详细信息
        if nce_layers is not None:
            print(f"\nDetailed NCE Layers (indices: {nce_layers}):")
            for idx in nce_layers:
                if idx < len(self.model):
                    layer = self.model[idx]
                    shape = shapes.get(idx, 'unknown')
                    print(f"  Layer {idx}: {layer.__class__.__name__} -> Output: {shape}")
                    # 如果是卷积层，打印更多细节
                    if hasattr(layer, 'weight') and layer.weight is not None:
                        print(f"            Weight shape: {layer.weight.shape}")
                else:
                    print(f"  WARNING: Layer {idx} out of range (total {len(self.model)} layers)")
            print("="*80)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        # Use non-inplace ReLU/LeakyReLU to avoid in-place modification issues
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(inplace=False)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    
netG_resnet = ResnetGenerator(input_nc=opt.input_nc, output_nc=1, ngf=64, n_blocks=6,
                        no_antialias=False, no_antialias_up=False)
nce_layers_resnet = [0,4,8,12,17]
netG_resnet.print_layer_info(input_shape=(1, opt.input_nc, opt.crop_size, opt.crop_size), 
                      nce_layers=nce_layers_resnet)