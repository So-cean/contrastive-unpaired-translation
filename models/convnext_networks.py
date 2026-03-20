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
    支持 channels_first 和 channels_last 的 LayerNorm
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
# ============================================
# ConvNeXt Block and ConvNeXt Generator
# ============================================

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt V1 Block
    结构: DWConv -> LayerNorm -> Linear(1x->4x) -> GELU -> Linear(4x->1x) -> LayerScale -> Residual
    
    输入输出: [N, C, H, W] (channels_first)
    内部处理: 转换为 [N, H, W, C] 进行 LayerNorm 和 Linear 操作
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                               padding=kernel_size//2, groups=dim)
        
        # 🔧 使用 nn.LayerNorm（作用在 channels_last 格式）
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer Scale: 可学习的逐通道缩放
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # 输入: [N, C, H, W]
        input = x
        x = self.dwconv(x)  # [N, C, H, W]
        
        # 转换为 channels_last: [N, C, H, W] -> [N, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # FFN（在 channels_last 格式下）
        x = self.norm(x)      # [N, H, W, C] - nn.LayerNorm 作用在最后一个维度
        x = self.pwconv1(x)   # [N, H, W, 4*C]
        x = self.act(x)       # [N, H, W, 4*C]
        x = self.pwconv2(x)   # [N, H, W, C]
        
        # Layer Scale（广播正确）
        if self.gamma is not None:
            x = self.gamma * x  # gamma: [C] 自动广播到 [N, H, W, C]
        
        # 转换回 channels_first: [N, H, W, C] -> [N, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # 残差连接
        x = input + self.drop_path(x)
        return x


class DropPath(nn.Module):
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

class ConvNeXtGenerator(nn.Module):
    """
    ConvNeXt V1 Generator for CUT/CycleGAN
    
    架构：
    - 输入层: 7x7 Conv + LayerNorm + GELU
    - 下采样: 2x (Conv + LayerNorm + GELU + Downsample)
    - ConvNeXt Blocks: n_blocks 个
    - 上采样: 2x (Upsample + Conv + LayerNorm + GELU)
    - 输出层: 7x7 Conv + Tanh
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None, 
                 use_dropout=False, n_blocks=6, padding_type='reflect', 
                 no_antialias=False, no_antialias_up=False, opt=None,
                 convnext_kernel_size=7, drop_path_rate=0.1,
                 layer_scale_init_value=1e-6):
        
        super(ConvNeXtGenerator, self).__init__()
        self.opt = opt
        self.n_blocks = n_blocks
        
        model = []
        
        # --- 入口 ---
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                  LayerNorm(ngf, eps=1e-6, data_format="channels_first"),
                  nn.GELU()]

        n_downsampling = 2
        
        # --- 下采样 ---
        for i in range(n_downsampling):
            mult = 2 ** i
            in_ch = ngf * mult
            out_ch = ngf * mult * 2
            
            if no_antialias:
                model += [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                          LayerNorm(out_ch, eps=1e-6, data_format="channels_first"),
                          nn.GELU()]
            else:
                model += [nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                          LayerNorm(in_ch, eps=1e-6, data_format="channels_first"), 
                          nn.GELU(),
                          Downsample(out_ch)]

        # --- ConvNeXt Blocks (核心) ---
        mult = 2 ** n_downsampling
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        
        for i in range(n_blocks):
            model += [ConvNeXtBlock(ngf * mult, 
                                   drop_path=dpr[i],
                                   layer_scale_init_value=layer_scale_init_value, 
                                   kernel_size=convnext_kernel_size)]

        # --- 上采样  ---
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            in_ch = ngf * mult
            out_ch = int(ngf * mult / 2)
            
            if no_antialias_up:
                # 标准 ConvTranspose
                model += [nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, 
                                            padding=1, output_padding=1),
                          LayerNorm(out_ch, eps=1e-6, data_format="channels_first"), 
                          nn.GELU()]
            else:
                model += [Upsample(in_ch),
                          nn.Conv2d(in_ch, out_ch, 3, padding=1),
                          LayerNorm(out_ch, eps=1e-6, data_format="channels_first"), 
                          nn.GELU()]

        # --- 出口 ---
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)
        
        # 权重初始化
        self.apply(self._init_weights)
        
        # 打印信息
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[ConvNeXtGenerator V1] n_blocks={n_blocks}, "
              f"kernel={convnext_kernel_size}, drop_path={drop_path_rate:.3f}, "
              f"layer_scale={layer_scale_init_value}, params={n_params:.2f}M")

    def _init_weights(self, m):
        """ConvNeXt 风格的权重初始化"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, input, layers=[], encode_only=False):
        """
        完全兼容 CUT 的多层特征提取接口
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
        
        
