#!/usr/bin/env python3
"""
Test script for MONAI dataset that outputs reconstructed NIfTI volumes.
This script processes 3D MRI volumes slice by slice and reconstructs complete volumes.
Uses the same data processing pipeline as the training dataset.
"""
import os
import sys
import torch
import numpy as np
import nibabel as nib
import glob
import random
from pathlib import Path
from collections import OrderedDict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from options.test_options import TestOptions
from models import create_model

try:
    import monai.transforms as monai_transforms
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not available. Please install MONAI for MRI data loading.")


class MonaiTestProcessor:
    """Mimics the MonaiDataset processing pipeline for testing complete volumes."""
    
    def __init__(self, opt):
        self.opt = opt        
        # Determine channel settings (same as training dataset)
        btoA = getattr(opt, 'direction', 'AtoB') == "BtoA"
        input_nc = getattr(opt, 'output_nc', 1) if btoA else getattr(opt, 'input_nc', 1)
        output_nc = getattr(opt, 'input_nc', 1) if btoA else getattr(opt, 'output_nc', 1)
        self.output_channels = max(input_nc, output_nc, 1)

        pixel_dim_opt = getattr(opt, 'pixel_dim')
        self.pix_dim = tuple(float(x) for x in pixel_dim_opt)

        print(f"MonaiTestProcessor using pix_dim: {self.pix_dim}")

        # Setup MONAI transforms - EXACTLY same as training dataset
        self.transform = monai_transforms.Compose([
            monai_transforms.LoadImaged(keys=["image"]),
            monai_transforms.EnsureChannelFirstd(keys=["image"]),
            monai_transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
            monai_transforms.Orientationd(keys=["image"], axcodes="RAS"),
            # Only resample x,y dimensions to 1mm, keep z as original
            monai_transforms.Spacingd(keys=["image"], pixdim=self.pix_dim, mode=("bilinear")),
            # Center crop and pad operations first - keep z dimension unchanged (-1)
            monai_transforms.CenterSpatialCropd(keys=["image"], roi_size=(256, 256, -1)),
            monai_transforms.SpatialPadd(keys=["image"], spatial_size=(256, 256, -1), mode="constant", constant_values=0),
            # Apply intensity normalization AFTER geometric transformations
            monai_transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1, clip=True),
        ])
    
    def process_volume_to_slices(self, nii_path, domain='A'):
        """
        Process entire volume and return all valid slices with their indices.
        This mimics the training dataset's slice extraction but returns ALL slices.
        """
        # Load and preprocess 3D volume
        processed_data = self.transform({"image": nii_path})  # dict with key "image"
        volume_tensor = processed_data['image'].squeeze(0)  # (H, W, D)
        
        # predict all slices in the volume
        processed_slices = []
        num_slices = volume_tensor.shape[-1]
        slice_indices = list(range(num_slices))
        for z in range(num_slices):
            slice_2d = volume_tensor[:, :, z]  # (H, W)
            # normalize slice to [-1, 1] as in training
            if self.output_channels == 1:
                slice_2d = slice_2d.unsqueeze(0)  # (1, 256, 256)
            else:
                slice_2d = slice_2d.unsqueeze(0).repeat(self.output_channels, 1, 1)
                
            slice_2d = slice_2d * 2.0 - 1.0
            processed_slices.append(slice_2d)    
        
        print(f"  Processed volume '{Path(nii_path).name}' to {num_slices} slices.")
        print(f"  Volume shape after processing: {volume_tensor.shape}")
        print(f"  processed image: {processed_data['image'].shape}")
        print(f"  processed_slices[0] shape: {processed_slices[0].shape}")
        
        return processed_slices, slice_indices, volume_tensor.shape, processed_data


class MonaiTester:
    def __init__(self, opt):
        self.opt = opt
        self.model = create_model(opt)
        self.model.setup(opt)
        
        # Create our custom processor
        self.processor = MonaiTestProcessor(opt)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Create output directory
        self.output_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

    def save_nifti(self, volume_data, original_nii, output_path, processed_data=None):
        """Save volume data as NIfTI file with correct spatial metadata."""
        
        # Create new header based on original
        new_header = original_nii.header.copy()
        new_header.set_data_shape(volume_data.shape)
        
        # Get correct spacing and affine from processed data
        if processed_data is not None:
            processed_image = processed_data["image"]
            
            # Use processed affine matrix (after MONAI transforms)
            if hasattr(processed_image, 'affine'):
                affine_matrix = processed_image.affine.numpy()
                print(f"Using processed affine from MONAI")
            else:
                affine_matrix = original_nii.affine
                print(f"Warning: No processed affine found, using original")
            
            # Use processed spacing (after MONAI transforms) or fallback to processor pix_dim
            if hasattr(processed_image, 'pixdim') and len(processed_image.pixdim) >= 3:
                processed_spacing = processed_image.pixdim[:3]
                print(f"Using processed spacing: {processed_spacing}")
            else:
                # Fallback: use the processor pix_dim's x,y and original z
                original_spacing = original_nii.header.get_zooms()
                processed_spacing = (self.processor.pix_dim[0], self.processor.pix_dim[1], original_spacing[2])
                print(f"Using calculated spacing from processor.pix_dim: {processed_spacing}")
            
            new_header.set_zooms(processed_spacing)
            
        else:
            # Fallback to original metadata
            affine_matrix = original_nii.affine
            original_spacing = original_nii.header.get_zooms()
            new_header.set_zooms(original_spacing)
            print(f"Warning: No processed data provided, using original metadata")
        
        # Create new NIfTI image with correct metadata
        new_nii = nib.Nifti1Image(volume_data, affine_matrix, new_header)
        
        # Copy important metadata from original
        new_nii.header['descrip'] = original_nii.header['descrip'] 
        new_nii.header['aux_file'] = original_nii.header['aux_file']
        
        nib.save(new_nii, output_path)
        print(f"Saved: {output_path}")
        print(f"  Final spacing: {new_nii.header.get_zooms()}")
        print(f"  Shape: {new_nii.shape}")
        print(f"  Affine diagonal: [{new_nii.affine[0,0]:.3f}, {new_nii.affine[1,1]:.3f}, {new_nii.affine[2,2]:.3f}]")

    def test_volumes(self):
        """Test volumes using custom processor with same pipeline as training dataset."""
        
        # Get volume paths for both domains
        source_dir_A = Path(self.opt.dataroot) / f"{self.opt.phase}A"
        source_dir_B = Path(self.opt.dataroot) / f"{self.opt.phase}B"
        
        volume_paths_A = sorted(list(source_dir_A.glob("*.nii.gz")))
        volume_paths_B = sorted(list(source_dir_B.glob("*.nii.gz")))
        
        if len(volume_paths_A) == 0:
            print(f"No .nii.gz files found in {source_dir_A}")
        if len(volume_paths_B) == 0:
            print(f"No .nii.gz files found in {source_dir_B}")
        
        if len(volume_paths_A) == 0 and len(volume_paths_B) == 0:
            print("No volumes found in either domain!")
            return
        
        # Process based on direction
        if self.opt.direction == 'AtoB':
            primary_paths = volume_paths_A[:self.opt.num_test] if volume_paths_A else []
            target_paths = volume_paths_B[:self.opt.num_test] if volume_paths_B else []
            primary_domain = 'A'
            target_domain = 'B'
            output_suffix = 'fake_B'
            real_suffix = 'real_A'
            target_suffix = 'real_B'
        else:
            primary_paths = volume_paths_B[:self.opt.num_test] if volume_paths_B else []
            target_paths = volume_paths_A[:self.opt.num_test] if volume_paths_A else []
            primary_domain = 'B'
            target_domain = 'A'
            output_suffix = 'fake_A'
            real_suffix = 'real_B'
            target_suffix = 'real_A'
        
        print(f"Processing {len(primary_paths)} volumes from domain {primary_domain}")
        print(f"Found {len(target_paths)} volumes in target domain {target_domain}")
        
        # Process primary domain volumes (source -> target harmonization)
        for i, volume_path in enumerate(primary_paths):
            print(f"\nProcessing volume {i+1}/{len(primary_paths)}: {Path(volume_path).name}")
            
            # Process volume to get all valid slices
            processed_slices, slice_indices, original_shape, processed_data = self.processor.process_volume_to_slices(
                str(volume_path), primary_domain
            )
            
            # Initialize output volumes
            fake_volume = torch.zeros(original_shape, dtype=torch.float32)
            real_volume = torch.zeros(original_shape, dtype=torch.float32)
            
            # Process each slice through the model
            for j, (slice_tensor, slice_idx) in enumerate(zip(processed_slices, slice_indices)):
                # Add batch dimension and move to device
                model_input = slice_tensor.unsqueeze(0).to(self.model.device)  # (1, C, 256, 256)
                
                # Run inference
                with torch.no_grad():
                    # Use CUT generator 'netG' for inference. Fall back to 'netG_A' if present for compatibility.
                    if hasattr(self.model, 'netG'):
                        gen = self.model.netG
                    elif hasattr(self.model, 'netG_A'):
                        gen = self.model.netG_A
                    else:
                        raise RuntimeError("Model does not expose a generator named 'netG' or 'netG_A'. Make sure --model cut is used.")

                    # move input to model device and run generator
                    fake_output = gen(model_input.to(self.model.device))

                    # Convert back to [0, 1]
                    fake_output = (fake_output + 1.0) / 2.0

                    if fake_output.shape[1] > 1:
                        fake_output = fake_output[:, 0:1, :, :]

                    fake_slice = fake_output.squeeze().cpu()  # (256, 256)

                    # Also convert real slice back to [0, 1] for saving
                    real_slice = (slice_tensor.squeeze() + 1.0) / 2.0  # (256, 256)

                # Store slices in their original positions
                fake_volume[:, :, slice_idx] = fake_slice
                real_volume[:, :, slice_idx] = real_slice

                if j % 20 == 0:
                    print(f"  Processed slice {j+1}/{len(processed_slices)} (z={slice_idx})")
            
            # Load original NIfTI for metadata
            original_nii = nib.load(volume_path)
            
            # Save fake volume (harmonization result) with correct processed metadata
            fake_filename = f"{Path(volume_path).with_suffix('').stem}_{output_suffix}.nii.gz"
            fake_output_path = self.output_dir / fake_filename

            # 应用mask：原图像>0的地方为1，否则为0
            mask = (real_volume.numpy() > 0).astype(fake_volume.numpy().dtype)
            fake_volume_masked = fake_volume.numpy() * mask
            self.save_nifti(fake_volume_masked, original_nii, str(fake_output_path), processed_data=processed_data)
            
            # Save real (preprocessed) volume for comparison with correct processed metadata
            real_filename = f"{Path(volume_path).with_suffix('').stem}_{real_suffix}.nii.gz"
            real_output_path = self.output_dir / real_filename
            self.save_nifti(real_volume.numpy(), original_nii, str(real_output_path), processed_data=processed_data)
        
        # Process target domain volumes (for reference/comparison)
        print(f"\nProcessing target domain volumes ({target_domain}) for reference...")
        for i, volume_path in enumerate(target_paths):
            if i >= self.opt.num_test:
                break
            
            print(f"Processing target volume {i+1}: {Path(volume_path).name}")
            
            # Process target domain volume (same preprocessing as training)
            processed_slices, slice_indices, original_shape, processed_data = self.processor.process_volume_to_slices(
                str(volume_path), target_domain
            )
            
            # Initialize target volume
            target_volume = torch.zeros(original_shape, dtype=torch.float32)
            
            # Convert processed slices back to [0, 1] for saving
            for j, (slice_tensor, slice_idx) in enumerate(zip(processed_slices, slice_indices)):
                target_slice = (slice_tensor.squeeze() + 1.0) / 2.0  # (256, 256)
                target_volume[:, :, slice_idx] = target_slice
            
            # Load original NIfTI for metadata
            original_nii = nib.load(volume_path)
            
            # Save target domain volume with correct processed metadata
            target_filename = f"{Path(volume_path).with_suffix('').stem}_{target_suffix}.nii.gz"
            target_output_path = self.output_dir / target_filename
            self.save_nifti(target_volume.numpy(), original_nii, str(target_output_path), processed_data=processed_data)
        
        print(f"\nProcessing completed! Results saved to: {self.output_dir}")
        print(f"Saved harmonization results ({output_suffix}), source domain ({real_suffix}), and target domain ({target_suffix}) volumes.")
        print("All volumes use the same preprocessing as training dataset.")


def main():
    # Parse options
    opt = TestOptions().parse()
    
    # Add device setting
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Ensure we're using the monai dataset
    if opt.dataset_mode != 'monai':
        print("Warning: This script is designed for --dataset_mode monai")
        opt.dataset_mode = 'monai'
    
    # Hard-code some parameters for test
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    
    if not MONAI_AVAILABLE:
        print("MONAI is required for this test script. Please install MONAI.")
        return
    
    # Create tester and run
    tester = MonaiTester(opt)
    tester.test_volumes()


if __name__ == "__main__":
    main()