#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import sys
import random
import glob
import numpy as np
import torch

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
from data.base_dataset import BaseDataset


import monai.transforms as monai_transforms
from monai.data import CacheDataset


class Monai3DDataset(BaseDataset):
    """
    This dataset class loads 3D MRI NIfTI files and returns 3D volumes for 3D CycleGAN training.

    It requires two directories to host training volumes:
    - trainA: thick slice data (厚层数据)
    - trainB: thin slice data (薄层数据)
    Each directory should contain .nii.gz files.

    For each 3D volume, the dataset returns the complete 3D volume for training.
    Both domains are processed using the same 3D method.
    Input NIfTI files are assumed to be isotropic.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add dataset-specific options for MONAI 3D dataset."""
        parser.add_argument('--pixel_dim', nargs=3, type=float, default=(1.0, 1.0, 1.0),
                            help='Pixel spacing passed to MONAI Spacingd. Provide three values: x y z (e.g. --pixel_dim 1.0 1.0 1.0).')
        parser.add_argument('--volume_size', nargs=3, type=int, default=(128, 128, 128),
                            help='Target volume size for 3D processing (e.g. --volume_size 128 128 128).')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # thick slice data
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # thin slice data

        # Load NIfTI file paths
        self.A_paths = sorted(glob.glob(os.path.join(self.dir_A, "*.nii.gz")))
        self.B_paths = sorted(glob.glob(os.path.join(self.dir_B, "*.nii.gz")))

        if opt.max_dataset_size != float("inf"):
            self.A_paths = self.A_paths[:opt.max_dataset_size]
            self.B_paths = self.B_paths[:opt.max_dataset_size]

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        if self.A_size == 0:
            raise ValueError("No .nii.gz files found in {}".format(self.dir_A))
        if self.B_size == 0:
            raise ValueError("No .nii.gz files found in {}".format(self.dir_B))

        print("Found {} files in domain A  and {} files in domain B ".format(self.A_size, self.B_size))

        # Determine if we should output grayscale or RGB based on opt settings
        btoA = getattr(opt, 'direction', 'AtoB') == "BtoA"
        input_nc = getattr(opt, 'output_nc', 1) if btoA else getattr(opt, 'input_nc', 1)
        output_nc = getattr(opt, 'input_nc', 1) if btoA else getattr(opt, 'output_nc', 1)

        # Use single channel for MRI data (grayscale)
        self.output_channels = max(input_nc, output_nc, 1)  # At least 1 channel

        print("Dataset will output {} channel(s) per volume".format(self.output_channels))

        pixel_dim_opt = getattr(opt, 'pixel_dim')
        pixel_dim = tuple(float(x) for x in pixel_dim_opt)

        volume_size_opt = getattr(opt, 'volume_size', (128, 128, 128))
        self.volume_size = tuple(int(x) for x in volume_size_opt)

        # Setup MONAI transforms - unified processing for both volume loading and 3D processing

        self.transform = monai_transforms.Compose([
            monai_transforms.LoadImaged(keys=["image"]),
            monai_transforms.EnsureChannelFirstd(keys=["image"]),
            monai_transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
            monai_transforms.Orientationd(keys=["image"], axcodes="RAS", labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
            monai_transforms.Spacingd(keys=["image"], pixdim=pixel_dim, mode=("bilinear")),
            monai_transforms.CenterSpatialCropd(keys=["image"], roi_size=self.volume_size),
            monai_transforms.SpatialPadd(keys=["image"], spatial_size=self.volume_size, mode="constant", constant_values=0),
            monai_transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
        ])
        self.data_A = CacheDataset(
            data=[{"image": path} for path in self.A_paths],
            transform=self.transform,
            cache_rate=1.0,
            num_workers=opt.num_threads,
            copy_cache=False,
        )
        self.data_B = CacheDataset(
            data=[{"image": path} for path in self.B_paths],
            transform=self.transform,
            cache_rate=1.0,
            num_workers=opt.num_threads,
            copy_cache=False,
        )

        # Initialize epoch counter for reproducible random sampling
        self.current_epoch = 0
        # Seed offset for independent sampling (used in SB model dual dataloaders)
        self.seed_offset = getattr(opt, 'seed_offset', 0)

    def set_epoch(self, epoch):
        """
        Set current epoch for reproducible random sampling across epochs.
        Call this method at the beginning of each epoch in your training loop.

        Parameters:
            epoch (int) -- current epoch number
        """
        self.current_epoch = epoch

    def load_volume(self, volume_idx, domain="A"):
        """Load 3D NIfTI volume (already cached via CacheDataset).
        Returns a torch.Tensor: 3D volume tensor (output_channels, H, W, D) normalized to [-1, 1]
        """
        data_domain = getattr(self, "data_{}".format(domain))
        volume_dict = data_domain[volume_idx]
        image = volume_dict["image"]  # (C, H, W, D) - may be torch.Tensor or numpy array

        volume_3d = image.as_tensor()

        volume_3d = volume_3d.repeat(self.output_channels, 1, 1, 1)  # (output_channels, H, W, D)
        volume_3d = volume_3d * 2.0 - 1.0  # Normalize to [-1, 1]

        return volume_3d

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int) -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) -- a 3D volume from domain A (thick slices) (output_channels, H, W, D)
            B (tensor) -- a 3D volume from domain B (thin slices) (output_channels, H, W, D)
            A_paths (str) -- NIfTI file paths
            B_paths (str) -- NIfTI file paths
        """
        # A: sequential access to ensure all A volumes are visited
        index_A = index % self.num_A_volumes

        # B: random sampling with reproducible seeds
        if self.opt.serial_batches:
            # For serial batches, use sequential access
            index_B = index % self.num_B_volumes
        else:
            # For random batches, use epoch + index + seed_offset as seed for reproducibility
            # This ensures:
            # 1. Same epoch + index -> same B volume (reproducible)
            # 2. Different epochs -> different B volumes (diversity across epochs)
            # 3. Different indices -> different B volumes (diversity within epoch)
            # 4. DDP-safe: all processes with same index get same B (if using DistributedSampler)
            # 5. Different seed_offset -> independent sampling (for SB model dual dataloaders)
            seed = self.current_epoch * 1000000 + index + self.seed_offset * 1000000000
            rng = np.random.RandomState(seed=seed)
            index_B = rng.randint(0, self.num_B_volumes)

        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]
        A = self.load_volume(index_A, domain='A')
        B = self.load_volume(index_B, domain='B')
        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    @property
    def num_A_volumes(self):
        return len(self.A_paths)
    @property
    def num_B_volumes(self):
        return len(self.B_paths)

    def __len__(self):
        return max(self.num_A_volumes, self.num_B_volumes)
