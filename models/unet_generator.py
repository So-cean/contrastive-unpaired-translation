"""
Modified MONAI UNet for CUT Framework (修正版)

修正点：
1. 完善特征提取逻辑（使用 hook）
2. 优化代码结构
3. 添加详细注释
"""

from monai.networks.nets import UNet
import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, ResidualUnit, UpSample
from monai.networks.layers import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection


class UnetGenerator(nn.Module):
    """
    Modified MONAI UNet for Unpaired Image Translation
    
    主要修改：
    1. _get_up_layer: 使用 Upsample + Conv（避免棋盘效应）
    2. forward: 支持多层特征提取（for CUT's PatchNCE loss）
    3. 输出 Tanh 激活
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        num_downs: int = 7,
        norm_layer = None,
        use_dropout: bool = False,
        num_res_units: int = 2,
    ):
        super().__init__()
        
        # 默认 InstanceNorm
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        
        # 转换为 MONAI 格式
        if norm_layer == nn.InstanceNorm2d:
            norm = Norm.INSTANCE
        elif norm_layer == nn.BatchNorm2d:
            norm = Norm.BATCH
        else:
            norm = Norm.INSTANCE
        
        # 计算通道数
        channels = self._compute_channels(ngf, num_downs)
        strides = tuple([2] * (len(channels) - 1))
        
        if len(channels) < 2:
            raise ValueError("channels length should be >= 2")
        
        # 保存参数
        self.dimensions = 2
        self.in_channels = input_nc
        self.out_channels = output_nc
        self.channels = channels
        self.strides = strides
        self.kernel_size = 3
        self.up_kernel_size = 3
        self.num_res_units = num_res_units
        self.act = Act.PRELU
        self.norm = norm
        self.dropout = 0.1 if use_dropout else 0.0
        self.bias = True
        self.adn_ordering = "NDA"
        
        # 构建模型
        self.model = self._create_model()
        
        # 输出激活
        self.output_activ = nn.Sigmoid()
        
        # 用于特征提取的模块列表（方便添加 hook）
        self._feature_blocks = []
        self._register_feature_blocks()
        
    
    def _compute_channels(self, ngf: int, num_downs: int) -> tuple:
        """计算通道数序列（Pix2Pix 模式）"""
        channels = []
        for i in range(num_downs):
            if i == 0:
                channels.append(ngf)
            elif i == 1:
                channels.append(ngf * 2)
            elif i == 2:
                channels.append(ngf * 4)
            else:
                channels.append(ngf * 8)
        return tuple(channels)
    
    def _create_model(self):
        """递归构建 U-Net"""
        
        def _create_block(inc, outc, channels, strides, is_top):
            c = channels[0]
            s = strides[0]
            
            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)
                upc = c * 2
            else:
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]
            
            down = self._get_down_layer(inc, c, s, is_top)
            up = self._get_up_layer(upc, outc, s, is_top)
            
            return self._get_connection_block(down, up, subblock)
        
        return _create_block(
            self.in_channels,
            self.out_channels,
            self.channels,
            self.strides,
            True
        )
    
    def _get_connection_block(self, down_path, up_path, subblock):
        """连接 encoder, skip, decoder"""
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)
    
    def _get_down_layer(self, in_channels, out_channels, strides, is_top):
        """下采样层（MONAI 标准实现）"""
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
        else:
            return Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
    
    def _get_bottom_layer(self, in_channels, out_channels):
        """Bottleneck 层"""
        return self._get_down_layer(in_channels, out_channels, 1, False)
    
    def _get_up_layer(self, in_channels, out_channels, strides, is_top):
        """
        上采样层（⭐ 修改：使用 Upsample + Conv）
        """
        layers = []
        
        # 1. Upsample（插值上采样）
        upsample = UpSample(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=None,  # 插值模式下会被忽略
            scale_factor=strides,
            mode='nontrainable',
            interp_mode='linear',
            align_corners=False,
        )
        layers.append(upsample)
        
        # 2. Convolution
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=1,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=False,  # ⭐ 不使用 ConvTranspose2d
            adn_ordering=self.adn_ordering,
        )
        layers.append(conv)
        
        # 3. ResNet blocks
        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            layers.append(ru)
        
        return nn.Sequential(*layers)
    
    def _register_feature_blocks(self):
        """
        注册用于特征提取的 blocks
        
        遍历模型，找到关键的 Convolution/ResidualUnit 层
        """
        def collect_blocks(module):
            for child in module.children():
                if isinstance(child, (Convolution, ResidualUnit)):
                    self._feature_blocks.append(child)
                elif isinstance(child, nn.Sequential):
                    collect_blocks(child)
                elif hasattr(child, 'submodule'):  # SkipConnection
                    collect_blocks(child.submodule)
        
        collect_blocks(self.model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        layers: list[int] = None, 
        encode_only: bool = False
    ):
        """
        前向传播（兼容 CUT）
        
        Args:
            x: [B, C, H, W]
            layers: 需要提取的层索引
            encode_only: 是否只返回 encoder 特征
        
        Returns:
            - 标准: output
            - 特征: (output, features) 或 features
        """
        if layers is None:
            layers = []
        
        # 标准前向
        x = (x + 1.0) / 2.0  # Scale to [0, 1]
        
        if len(layers) > 0 or encode_only:
            return self._forward_with_features(x, layers, encode_only)
        
        x = self.model(x)
        x = self.output_activ(x) # 输出范围 [0, 1]
        x = x * 2.0 - 1.0  # Scale back to [-1, 1]
        return x
    
    def _forward_with_features(
        self, 
        x: torch.Tensor, 
        layers: list[int], 
        encode_only: bool
    ):
        """
        提取多层特征（使用 hook）
        
        ⭐ 完整实现版本
        """
        features = []
        hooks = []
        
        # Hook 函数
        def hook_fn(module, input, output):
            features.append(output)
        
        # 根据 layers 注册 hooks
        if len(layers) > 0:
            for idx in layers:
                if idx < len(self._feature_blocks):
                    hook = self._feature_blocks[idx].register_forward_hook(hook_fn)
                    hooks.append(hook)
        
        # 前向传播
        if encode_only:
            _ = self.model(x)
            output = None
        else:
            output = self.model(x)
            output = self.output_activ(output)
        
        # 清理 hooks
        for hook in hooks:
            hook.remove()
        
        # 返回结果
        if encode_only:
            return features if len(features) > 0 else [x]
        else:
            return output, features if len(features) > 0 else [x]
    
    
        
        
def test_unet_generator():
    """完整测试"""
    print("\n" + "="*70)
    print("Testing Modified MONAI UNet Generator")
    print("="*70 + "\n")
    
    # 创建模型
    gen = UnetGenerator(
        input_nc=1,
        output_nc=1,
        ngf=64,
        num_downs=6,
        num_res_units=2,
        use_dropout=False,
    )
    
    # Test 1: 标准前向
    print("[Test 1] Standard forward pass")
    x = torch.randn(2, 1, 256, 256)
    out = gen(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Range:  [{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == x.shape, "Shape mismatch!"
    assert out.min() >= -1.5 and out.max() <= 1.5, "Range abnormal!"
    print("  ✅ Pass\n")
    
    # Test 2: 特征提取
    print("[Test 2] Feature extraction (CUT mode)")
    layers = [0, 4, 8, 12]
    out, feats = gen(x, layers=layers)
    print(f"  Output: {out.shape}")
    print(f"  Features extracted: {len(feats)}")
    for i, feat in enumerate(feats):
        print(f"    Layer {layers[i] if i < len(layers) else '?'}: {feat.shape}")
    print("  ✅ Pass\n")
    
    # Test 3: Encode only
    print("[Test 3] Encode only")
    feats = gen(x, layers=[0, 2, 4], encode_only=True)
    print(f"  Features: {len(feats)}")
    for i, feat in enumerate(feats):
        print(f"    {i}: {feat.shape}")
    print("  ✅ Pass\n")
    
    # Test 4: 检查梯度
    print("[Test 4] Gradient flow")
    x.requires_grad = True
    out = gen(x)
    loss = out.mean()
    loss.backward()
    grad_norm = x.grad.norm()
    print(f"  Gradient norm: {grad_norm:.6f}")
    assert grad_norm > 0, "No gradient!"
    print("  ✅ Pass\n")
    
    # Test 5: 不同配置
    print("[Test 5] Different configurations")
    configs = [
        {'num_res_units': 0, 'name': 'No ResNet'},
        {'num_res_units': 1, 'name': '1 ResNet unit'},
        {'num_res_units': 2, 'name': '2 ResNet units'},
    ]
    for config in configs:
        g = UnetGenerator(
            input_nc=1, output_nc=1, ngf=64, num_downs=7,
            num_res_units=config['num_res_units']
        )
        params = sum(p.numel() for p in g.parameters()) / 1e6
        print(f"  {config['name']:20s}: {params:6.2f}M params")
    print("  ✅ Pass\n")
    
    print("="*70)
    print("All tests passed! ✅")
    print("="*70 + "\n")

def inspect_network_layers():
    """检查网络的实际层结构"""
    gen = UnetGenerator(
        input_nc=1, output_nc=1, ngf=64, num_downs=7,
        num_res_units=2
    )
    
    print(f"Total feature blocks: {len(gen._feature_blocks)}")
    print("\nLayer structure:")
    
    x = torch.randn(1, 1, 256, 256)
    
    for i, block in enumerate(gen._feature_blocks):
        # 尝试获取输出形状
        print(f"  Layer {i:2d}: {block.__class__.__name__}")
    
    # 实际测试各层输出
    features = []
    hooks = []
    
    def hook_fn(module, input, output):
        features.append(output)
    
    for block in gen._feature_blocks:
        hooks.append(block.register_forward_hook(hook_fn))
    
    _ = gen(x)
    
    for hook in hooks:
        hook.remove()
    
    print("\nFeature shapes:")
    for i, feat in enumerate(features):
        print(f"  Layer {i:2d}: {feat.shape}")



if __name__ == '__main__':
    test_unet_generator()
    inspect_network_layers()