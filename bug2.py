import torch
import torch.nn.functional as F
from models import create_model
from options.test_options import TestOptions
import monai.transforms as monai_transforms

import importlib
import shutil
import os
if shutil.which("npu-smi") and importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    os.environ['HCCL_EXEC_TIMEOUT'] = '120'  
    os.environ['HCCL_CONNECT_TIMEOUT'] = '120'
    
opt = TestOptions().parse()
model = create_model(opt)
# load model
model.setup(opt)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = monai_transforms.Compose([
        monai_transforms.LoadImaged(keys=["image"]),
        monai_transforms.EnsureChannelFirstd(keys=["image"]),
        monai_transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
        monai_transforms.Orientationd(keys=["image"], axcodes="RAS"),
        # Only resample x,y dimensions to 1mm, keep z as original
        monai_transforms.Spacingd(keys=["image"], pixdim=(1,1,-1), mode=("bilinear")),
        # Center crop and pad operations first - keep z dimension unchanged (-1)
        monai_transforms.CenterSpatialCropd(keys=["image"], roi_size=(256, 256, -1)),
        monai_transforms.SpatialPadd(keys=["image"], spatial_size=(256, 256, -1), mode="constant", constant_values=0),
        # Apply intensity normalization AFTER geometric transformations
        monai_transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
    ])

nii_path = "/public/home_data/home/songhy2024/data/PVWMI/T1w/skullstripped/K_010_t1_skullstripped.nii.gz"
data = transform({"image": nii_path})
real_A = data["image"].unsqueeze(0)  # 添加 batch 维度
z = real_A.shape[-1]
real_A = real_A[..., z//2]  # 取中间切片进行测试 [1, C, H, W]
real_A = real_A * 2 - 1  # 归一化到 [-1, 1]
real_A = real_A.to(device)

# real_A status
print(f"real_A: min={real_A.min():.4f}, max={real_A.max():.4f}, mean={real_A.mean():.4f}, std={real_A.std():.4f}")
with torch.no_grad():
    # 获取多层特征
    layers = range(20)
    feats = model.netG(real_A, layers=layers, encode_only=True)
    
    for i, feat in enumerate(feats):
        B, C, H, W = feat.shape
        
        # 下采样 mask 到 feature 尺寸
        mask = F.interpolate(real_A, size=(H, W), mode='nearest')
        bg_mask = (mask <= -0.999).float()  # 纯背景
        fg_mask = (mask > -0.999).float()   # 前景
        
        # 计算背景和前景的 feature 统计
        for b in range(B):
            feat_b = feat[b]  # [C, H, W]
            
            bg_feats = feat_b * bg_mask[b]  # 背景区域
            fg_feats = feat_b * fg_mask[b]  # 前景区域
            
            bg_mean = bg_feats.sum() / (bg_mask[b].sum() + 1e-6)
            fg_mean = fg_feats.sum() / (fg_mask[b].sum() + 1e-6)
            
            feat_min = feat_b.min()
            feat_max = feat_b.max()
            feat_avg = feat_b.mean()            
            
            print(f"Layer {i}: bg_mean={bg_mean:.4f}, fg_mean={fg_mean:.4f}, min={feat_min:.4f}, max={feat_max:.4f}, avg={feat_avg:.4f}")

'''
python bug2.py \
    --dataroot /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
    --results_dir /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_npu_pred_convnext2/ \
    --name CUT_monai_K2E2_npu_convnext2 --model cut --direction AtoB --no_dropout --no_flip --dataset_mode monai \
    --netG convnext_24blocks \
    --nce_layers 1,8,14,23,30,35 \
    --batch_size 16 --input_nc 1 --output_nc 1 --preprocess scale_width_and_crop --ngf 64
    
python bug2.py \
    --dataroot /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
    --results_dir /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_npu_pred_min/ \
    --name CUT_monai_K2E2_npu_min --model cut --direction AtoB --no_dropout --no_flip --dataset_mode monai \
    --batch_size 16 --input_nc 1 --output_nc 1 --preprocess scale_width_and_crop --ngf 96 --phase all
'''