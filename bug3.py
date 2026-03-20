# test_multiscale_d.py

import torch
import sys
sys.path.append('.')

from models.networks import MultiScaleDiscriminator
import torch.nn as nn

print("="*70)
print("Testing MultiScaleDiscriminator")
print("="*70)

# 创建多尺度 D
disc = MultiScaleDiscriminator(
    input_nc=1,
    ndf=64,
    n_layers=3,
    norm_layer=nn.InstanceNorm2d,
    num_D=3,
    no_antialias=False
)

print(f"\nCreated MultiScaleDiscriminator:")
print(f"  num_D: 3")
print(f"  n_layers: 3")
print(f"  ndf: 64")

# 测试输入
x = torch.randn(2, 1, 256, 256)  # batch_size=2
print(f"\nInput shape: {x.shape}")

# 前向传播
print("\nRunning forward pass...")
with torch.no_grad():
    outputs = disc(x)

print(f"\nOutput type: {type(outputs)}")
print(f"Number of scales: {len(outputs)}")

# 详细输出
print("\n" + "-"*70)
print("Detailed Output Information")
print("-"*70)

for i, output in enumerate(outputs):
    input_size = 256 // (2 ** i)
    rf_on_input = 70
    rf_on_original = rf_on_input * (2 ** i)
    
    print(f"\nDiscriminator {i}:")
    print(f"  Input size:        {input_size}×{input_size}")
    print(f"  Output shape:      {output.shape}")
    print(f"  Output spatial:    {output.shape[2]}×{output.shape[3]}")
    print(f"  Receptive field:   ~{rf_on_original}×{rf_on_original} (on 256×256)")
    print(f"  Coverage:          ~{min(100, (rf_on_original/256)**2 * 100):.1f}%")
    print(f"  Total patches:     {output.shape[2] * output.shape[3]}")

# 测试 loss 计算
print("\n" + "-"*70)
print("Testing Loss Computation")
print("-"*70)

try:
    import torch.nn.functional as F
    
    # 模拟 GAN loss
    total_loss = 0.0
    for i, output in enumerate(outputs):
        target = torch.ones_like(output)
        loss = F.binary_cross_entropy_with_logits(output, target)
        total_loss += loss
        print(f"  D_{i} loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(outputs)
    print(f"\n  Average loss: {avg_loss.item():.4f}")
    print("\n✅ Loss computation successful!")
    
except Exception as e:
    print(f"\n❌ Error in loss computation: {e}")

print("\n" + "="*70)
print("Test completed successfully!")
print("="*70)