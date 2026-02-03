import os
import sys
import random
import glob
import numpy as np
import nibabel as nib
import torch
from collections import OrderedDict
import torch.distributed as dist

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.base_dataset import BaseDataset


import monai.transforms as monai_transforms
from monai.data import CacheDataset


class MonaiDataset(BaseDataset):
    """
    This dataset class loads 3D MRI NIfTI files and returns 2D slices for CycleGAN training.
    
    It requires two directories to host training volumes:
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
            
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  

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
            
        print(f"Found {self.A_size} files in domain A  and {self.B_size} files in domain B ")

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
        
        # Calculate total valid slices for each domain
        self._calculate_valid_slices(domain='A')
        self._calculate_valid_slices(domain='B')        
        
        self.epoch_A_indices = None
        
    def set_epoch(self, epoch):
        """每个 epoch 重新随机配对，确保 A 和 B 都不重复"""
        self.current_epoch = epoch
    
        g = torch.Generator()
        g.manual_seed(epoch)
    
        # A 多：随机选 num_B 个不重复的 A
        self.epoch_A_indices = torch.randperm(
            self.num_A_slices, generator=g
        )[:self.num_B_slices].tolist()
            
    def _calculate_valid_slices(self, domain:str='A'):
        """Calculate the total number of valid slices for each domain."""
        data_domain = getattr(self, f"data_{domain}")        
        min_nonzero_pixels = 100  # Minimum number of non-zero pixels to consider a slice valid
        slice_list = []
        slices_per_volume = {i: [] for i in range(len(data_domain))}
        
        for volume_idx in range(len(data_domain)):
            volume_dict = data_domain[volume_idx]
            data = volume_dict["image"] # (C, H, W, D)
            
            valid_z = [z for z in range(data.shape[-1]) if np.count_nonzero(data[..., z]) > min_nonzero_pixels]
            
            if not valid_z:
                valid_z = [data.shape[-1] // 2]
                
            slice_list.extend([(volume_idx, z) for z in valid_z])
            slices_per_volume[volume_idx] = valid_z
        
        setattr(self,f"slice_list_{domain}",slice_list)
        setattr(self, f"slices_per_volume_{domain}", slices_per_volume)
        
        print(f"Domain {domain} total valid slices: {len(slice_list)}")

    def load_and_extract_slice(self, volume_idx, z, domain:str="A"):
        """Load 3D NIfTI volume (already cached via CacheDataset) and extract the specified 2D slice.
        Returns a torch.Tensor: 2D slice tensor (output_channels, 256, 256) normalized to [-1, 1]
        """
        data_domain = getattr(self, f"data_{domain}")
        volume_dict = data_domain[volume_idx]
        image = volume_dict["image"]  # (C, H, W, D) - may be torch.Tensor or numpy array

        slice_2d = image[..., z].as_tensor()

        slice_2d = slice_2d.repeat(self.output_channels, 1, 1)  # (output_channels, 256, 256)
        slice_2d = slice_2d * 2.0 - 1.0  # Normalize to [-1, 1]      
        
        return slice_2d        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int) -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) -- an image slice from domain A  (output_channels, 256, 256)
            B (tensor) -- an image slice from domain B  (output_channels, 256, 256)
            A_paths (str) -- NIfTI file paths
            B_paths (str) -- NIfTI file paths
        """       
        # make sure num_A_slices is more than num_B_slices
        # B 确定（不重复）
        index_B = index
        index_A = self.epoch_A_indices[index]
            
        volume_idx_A, z_A = getattr(self, f"slice_list_A")[index_A]
        volume_idx_B, z_B = getattr(self, f"slice_list_B")[index_B]

        A_path = self.A_paths[volume_idx_A]
        B_path = self.B_paths[volume_idx_B]
        A = self.load_and_extract_slice(volume_idx_A, z_A, domain='A')
        B = self.load_and_extract_slice(volume_idx_B, z_B, domain='B')
        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    @property
    def num_A_slices(self):
        return len(getattr(self, "slice_list_A", []))
    @property
    def num_B_slices(self):
        return len(getattr(self, "slice_list_B", []))
    @property
    def num_A_volumes(self):
        return len(self.A_paths)
    @property
    def num_B_volumes(self):
        return len(self.B_paths)
    
    def __len__(self):
        return min(self.num_A_slices, self.num_B_slices)
    
if __name__ == "__main__":
    # Simple test to verify dataset functionality
    class DummyOpt:
        def __init__(self):
            # self.dataroot = "/public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2D-SIEMENS-AERA-1.5T/"
            self.dataroot = "/public/home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/"
            # self.csv_path = "/public_bme2/bme-wangqian2/songhy2024/harmonization_mri/dataset/PVWMI_data.csv"
            self.phase = "train"
            self.max_dataset_size = float("inf")
            self.pixel_dim = (1.0, 1.0, -1)
            self.num_threads = 4
            self.serial_batches = False
            self.direction = 'AtoB'
            self.input_nc = 1
            self.output_nc = 1
            
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