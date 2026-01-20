import os
import re
import sys
import random
import glob
import numpy as np
import torch
from collections import OrderedDict
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.base_dataset import BaseDataset

try:
    import monai
    import monai.transforms as monai_transforms
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not available. Please install MONAI for MRI data loading.")


class MonaiDataset(BaseDataset):
    """
    This dataset class loads 3D MRI NIfTI files and returns 2D slices for CycleGAN training.
    
    It requires two directories to host training volumes:
    - trainA: thick slice data (厚层数据)
    - trainB: thin slice data (薄层数据) 
    Each directory should contain .nii.gz files.
    
    For each 3D volume, the dataset randomly selects a 2D slice for training.
    Both domains are processed using the same 2D method.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add dataset-specific options for MONAI dataset."""
        parser.add_argument('--pixel_dim', nargs=3, type=float, default=(1.0, 1.0, -1),
                            help='Pixel spacing passed to MONAI Spacingd. Provide three values: x y z (e.g. --pixel_dim 0.8 0.8 -1).')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is required for MRI dataset. Please install with: pip install monai")
            
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
            raise ValueError(f"No .nii.gz files found in {self.dir_A}")
        if self.B_size == 0:
            raise ValueError(f"No .nii.gz files found in {self.dir_B}")
            
        print(f"Found {self.A_size} files in domain A (thick slices) and {self.B_size} files in domain B (thin slices)")
        
        # Initialize cache for loaded volumes
        self.cache_size = getattr(opt, 'cache_size', 100)  # Cache up to 200 volumes by default
        self.volume_cache = OrderedDict()  # LRU cache
        print(f"Volume cache size: {self.cache_size}")
        
        # Determine if we should output grayscale or RGB based on opt settings
        btoA = getattr(opt, 'direction', 'AtoB') == "BtoA"
        input_nc = getattr(opt, 'output_nc', 1) if btoA else getattr(opt, 'input_nc', 1)
        output_nc = getattr(opt, 'input_nc', 1) if btoA else getattr(opt, 'output_nc', 1)
        
        # Use single channel for MRI data (grayscale)
        self.output_channels = max(input_nc, output_nc, 1)  # At least 1 channel
        
        print(f"Dataset will output {self.output_channels} channel(s) per slice")
        
        pixel_dim_opt = getattr(opt, 'pixel_dim')
        pixel_dim = tuple(float(x) for x in pixel_dim_opt)

        print(f"Using pixel_dim (for Spacingd): {pixel_dim}")

        # Setup MONAI transforms - unified processing for both volume loading and slice processing
        
        self.transform = monai_transforms.Compose([
            monai_transforms.LoadImaged(keys=["image"]),
            monai_transforms.EnsureChannelFirstd(keys=["image"]),
            monai_transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
            monai_transforms.Orientationd(keys=["image"], axcodes="RAS"),
            monai_transforms.Spacingd(keys=["image"], pixdim=pixel_dim, mode=("bilinear")),
            monai_transforms.CenterSpatialCropd(keys=["image"], roi_size=(256, 256, -1)),
            monai_transforms.SpatialPadd(keys=["image"], spatial_size=(256, 256, -1), mode="constant", constant_values=0),
            monai_transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1, clip=True),
        ])
        
        
        # Calculate total valid slices for each domain
        self._calculate_valid_slices()
        
    
    def _get_cached_volume(self, nii_path):
        """Get volume from cache or load and cache it."""
        if nii_path in self.volume_cache:
            # Move to end (most recently used)
            volume = self.volume_cache.pop(nii_path)
            self.volume_cache[nii_path] = volume
            return volume
        
        # Load volume and add to cache - use dictionary format for MONAI transforms with 'd' suffix
        data_dict = {"image": nii_path}
        processed_dict = self.transform(data_dict)
        volume_tensor = processed_dict["image"]
        
        # Remove oldest item if cache is full
        if len(self.volume_cache) >= self.cache_size:
            self.volume_cache.popitem(last=False)  # Remove least recently used
        
        self.volume_cache[nii_path] = volume_tensor
        return volume_tensor

    def _calculate_valid_slices(self):
        """Calculate the total number of valid slices for each domain."""
        print("Calculating valid slices for each domain...")
        
        self.A_total_slices = 0
        self.B_total_slices = 0
        
        # Calculate for domain A (thick slices)
        for path in self.A_paths:
            try:
                volume_tensor = self._get_cached_volume(path)
                if volume_tensor.dim() == 4:
                    volume_tensor = volume_tensor.squeeze(0)
                
                if volume_tensor.dim() == 3:
                    _, _, depth = volume_tensor.shape
                    # Count valid slices
                    valid_count = 0
                    for i in range(depth):
                        slice_2d = volume_tensor[:, :, i]
                        if torch.sum(slice_2d > 0.1) > (slice_2d.numel() * 0.05):
                            valid_count += 1
                    
                    # If no valid slices, use middle 50% as fallback
                    if valid_count == 0:
                        valid_count = max(1, depth // 2)
                    
                    self.A_total_slices += valid_count
            except Exception as e:
                print(f"Warning: Could not process {path}: {e}")
                self.A_total_slices += 1
        
        # Calculate for domain B (thin slices)
        for path in self.B_paths:
            try:
                volume_tensor = self._get_cached_volume(path)
                if volume_tensor.dim() == 4:
                    volume_tensor = volume_tensor.squeeze(0)
                
                if volume_tensor.dim() == 3:
                    _, _, depth = volume_tensor.shape
                    # Count valid slices
                    valid_count = 0
                    for i in range(depth):
                        slice_2d = volume_tensor[:, :, i]
                        if torch.sum(slice_2d > 0.1) > (slice_2d.numel() * 0.05):
                            valid_count += 1
                    
                    # If no valid slices, use middle 50% as fallback
                    if valid_count == 0:
                        valid_count = max(1, depth // 2)
                    
                    self.B_total_slices += valid_count
            except Exception as e:
                print(f"Warning: Could not process {path}: {e}")
                self.B_total_slices += 1
        
        print(f"Domain A (thick) total valid slices: {self.A_total_slices}")
        print(f"Domain B (thin) total valid slices: {self.B_total_slices}")

    def load_and_extract_slice(self, nii_path, domain='A'):
        """Load 3D NIfTI volume and extract a random 2D slice.
        
        Parameters:
            nii_path (str): Path to the NIfTI file
            domain (str): Domain identifier ('A' for thick, 'B' for thin)
            
        Returns:
            torch.Tensor: 2D slice tensor (output_channels, 256, 256) normalized to [-1, 1]
        """
        # Load and preprocess 3D volume using cached MONAI transform
        volume_tensor = self._get_cached_volume(nii_path)
        
        # Remove channel dimension for slice selection (C, H, W, D) -> (H, W, D)
        if volume_tensor.dim() == 4:
            volume_tensor = volume_tensor.squeeze(0)
        
        # Get volume dimensions
        if volume_tensor.dim() == 3:
            height, width, depth = volume_tensor.shape
        else:
            raise ValueError(f"Unexpected tensor dimensions: {volume_tensor.shape}")
        
        # Find valid slices (non-empty slices)
        valid_slices = []
        for i in range(depth):
            slice_2d = volume_tensor[:, :, i]
            # Check if slice has meaningful content (not just zeros/background)
            if torch.sum(slice_2d > 0.1) > (slice_2d.numel() * 0.05):  # At least 5% non-background
                valid_slices.append(i)
        
        # If no valid slices found, use middle slices as fallback
        if len(valid_slices) == 0:
            # Use middle 50% of slices as fallback
            start_idx = depth // 4
            end_idx = 3 * depth // 4
            valid_slices = list(range(max(0, start_idx), min(depth, end_idx)))
        
        # Randomly select from valid slices
        slice_idx = random.choice(valid_slices)
        
        # Extract 2D slice (already processed to 256x256 by the unified transform)
        slice_2d = volume_tensor[:, :, slice_idx]  # (256, 256)
        
        # Convert to the appropriate number of channels
        if self.output_channels == 1:
            # Keep as single channel (grayscale)
            slice_2d = slice_2d.unsqueeze(0)  # (1, 256, 256)
        else:
            # Repeat to create RGB channels
            slice_2d = slice_2d.unsqueeze(0).repeat(self.output_channels, 1, 1)  # (output_channels, 256, 256)
        
        # Normalize to [-1, 1] range (standard for GAN training)
        slice_2d = slice_2d * 2.0 - 1.0
        
        return slice_2d

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int) -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) -- an image slice from domain A (thick slices) (output_channels, 256, 256)
            B (tensor) -- an image slice from domain B (thin slices) (output_channels, 256, 256)
            A_paths (str) -- NIfTI file paths
            B_paths (str) -- NIfTI file paths
        """
        # A域按顺序取，B域用取模或随机采样
        A_path = self.A_paths[index % self.A_size]
        # B域用取模或随机采样
        if self.B_size < self.A_size:
            B_index = index % self.B_size
        else:
            B_index = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[B_index]
        A = self.load_and_extract_slice(A_path, 'A')
        B = self.load_and_extract_slice(B_path, 'B')
        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        # 返回最大slice数，保证所有slice都能用上
        return max(self.A_total_slices, self.B_total_slices)
    
if __name__ == "__main__":
    # Simple test to verify dataset functionality
    class DummyOpt:
        def __init__(self):
            # self.dataroot = "/public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2D-SIEMENS-AERA-1.5T/"
            self.dataroot = "/public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/"
            # self.csv_path = "/public_bme2/bme-wangqian2/songhy2024/harmonization_mri/dataset/PVWMI_data.csv"
            self.phase = "train"
            self.max_dataset_size = float("inf")
            
    opt = DummyOpt()
    dataset = MonaiDataset(opt)
    print(f"Dataset size: {len(dataset)}")
    
    try:
        sample = dataset[0]
        print(f"A shape: {sample['A'].shape}, B shape: {sample['B'].shape}")
        print(f"A range: [{sample['A'].min():.3f}, {sample['A'].max():.3f}]")
        print(f"B range: [{sample['B'].min():.3f}, {sample['B'].max():.3f}]")
        print(f"A path: {sample['A_paths']}")
        print(f"B path: {sample['B_paths']}")
        print("Test successful!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    # vis 3 samples
    import matplotlib.pyplot as plt
    for i in range(3):
        sample = dataset[i]
        A_img = (sample['A'].numpy().transpose(1, 2, 0) + 1) / 2  # Convert back to [0, 1] for visualization
        B_img = (sample['B'].numpy().transpose(1, 2, 0) + 1) / 2
        
        plt.subplot(2, 3, i + 1)
        plt.imshow(A_img)
        plt.axis('off')
        
        plt.subplot(2, 3, i + 1 + 3)
        plt.imshow(B_img)
        plt.axis('off')
        
    plt.show()
    plt.tight_layout()
    # save figure
    plt.savefig("unaligned_dataset_mri_samples.png", dpi=300)