"""The shared 2D MRI geometry/intensity pipeline for training and inference."""
import torch
from monai import transforms as T


def volume_transform(pixel_dim):
    return T.Compose([
        T.LoadImaged(keys=['image']),
        T.EnsureChannelFirstd(keys=['image']),
        T.EnsureTyped(keys=['image'], dtype=torch.float32),
        T.Orientationd(keys=['image'], axcodes='RAS', labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
        T.Spacingd(keys=['image'], pixdim=tuple(pixel_dim), mode='bilinear'),
        T.CenterSpatialCropd(keys=['image'], roi_size=(256, 256, -1)),
        T.SpatialPadd(keys=['image'], spatial_size=(256, 256, -1), mode='constant', constant_values=0),
        T.ScaleIntensityRangePercentilesd(keys=['image'], lower=0.5, upper=99.5,
                                        b_min=0, b_max=1, clip=True),
    ])
