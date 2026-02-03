import os
import numpy as np
import nibabel as nib
import pandas as pd
from collections import defaultdict
import glob

def auto_discover_dataset_structure(dataroot):
    """
    自动发现并验证标准的CycleGAN文件夹结构
    Expected structure:
    dataroot/
    ├── trainA/
    ├── trainB/
    ├── valA/
    ├── valB/
    ├── testA/
    └── testB/
    """
    expected_folders = ['trainA', 'trainB', 'valA', 'valB', 'testA', 'testB']
    dataset_structure = {}
    
    print("正在扫描数据集结构...")
    for folder in expected_folders:
        folder_path = os.path.join(dataroot, folder)
        if os.path.exists(folder_path):
            # 查找所有.nii和.nii.gz文件
            nii_files = glob.glob(os.path.join(folder_path, "*.nii"))
            nii_gz_files = glob.glob(os.path.join(folder_path, "*.nii.gz"))
            all_files = nii_files + nii_gz_files
            
            if all_files:
                dataset_structure[folder] = all_files
                print(f"  ✓ {folder}: 找到 {len(all_files)} 个文件")
            else:
                print(f"  ⚠️ {folder}: 文件夹存在但未找到.nii文件")
                dataset_structure[folder] = []
        else:
            print(f"  ❌ {folder}: 文件夹不存在")
            dataset_structure[folder] = []
    
    # 检查是否至少找到了一个域的数据
    valid_domains = [k for k, v in dataset_structure.items() if v]
    if not valid_domains:
        raise ValueError(f"在 {dataroot} 中未找到任何有效的NIfTI文件")
    
    print(f"扫描完成！找到了 {len(valid_domains)} 个有效的数据域")
    return dataset_structure

def get_nifti_z_spacing(file_path):
    """获取NIfTI文件的z-spacing"""
    try:
        img = nib.load(file_path)
        pixdim = img.header.get_zooms()
        return pixdim[2]  # 返回z轴间距
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

def get_nifti_slice_count(file_path):
    """获取NIfTI文件的切片数量（z轴维度）"""
    try:
        img = nib.load(file_path)
        return img.shape[2]  # 假设z轴是第3个维度 (x, y, z)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return 0

def analyze_dataset(dataroot):
    """
    主分析函数：自动扫描目录并分析NIfTI数据集
    """
    # 1. 自动发现数据集结构
    dataset_structure = auto_discover_dataset_structure(dataroot)
    
    # 2. 分析每个文件
    stats = defaultdict(lambda: defaultdict(list))
    
    for part, file_list in dataset_structure.items():
        if not file_list:
            continue
            
        print(f"\n分析 {part}...")
        for file_path in file_list:
            z_spacing = get_nifti_z_spacing(file_path)
            slice_count = get_nifti_slice_count(file_path)
            
            if z_spacing is not None and slice_count > 0:
                stats[part]['z_spacings'].append(z_spacing)
                stats[part]['slice_counts'].append(slice_count)
                stats[part]['file_paths'].append(os.path.basename(file_path))
    
    return stats, dataset_structure

def print_statistics(stats):
    """打印详细的统计结果"""
    print("\n" + "="*80)
    print("NIfTI数据集统计分析结果")
    print("="*80)
    
    results = []
    for part in ['trainA', 'valA', 'testA', 'trainB', 'valB', 'testB']:
        if part not in stats or not stats[part]['z_spacings']:
            print(f"\n{part}: 无有效数据")
            continue
        
        z_spacings = stats[part]['z_spacings']
        slice_counts = stats[part]['slice_counts']
        
        # 计算统计量
        avg_z_spacing = np.mean(z_spacings)
        std_z_spacing = np.std(z_spacings)
        avg_slices = np.mean(slice_counts)
        total_volumes = len(z_spacings)
        total_slices = np.sum(slice_counts)
        
        print(f"\n{part.upper()} 统计:")
        print(f"  体积数量: {total_volumes}")
        print(f"  Z-spacing平均: {avg_z_spacing:.4f} ± {std_z_spacing:.4f} mm")
        print(f"  平均切片数/体积: {avg_slices:.1f}")
        print(f"  总切片数: {total_slices}")
        print(f"  Z-spacing范围: [{min(z_spacings):.4f}, {max(z_spacings):.4f}]")
        print(f"  切片数范围: [{min(slice_counts)}, {max(slice_counts)}]")
        
        results.append({
            'Part': part, 'Volumes': total_volumes,
            'Avg_Z_Spacing(mm)': f"{avg_z_spacing:.4f} ± {std_z_spacing:.4f}",
            'Avg_Slices_Per_Volume': f"{avg_slices:.1f}",
            'Total_Slices': total_slices
        })
    
    return results

# 使用示例
if __name__ == "__main__":
    # 只需要提供数据根目录路径
    dataroot = "/public/home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/"  # 请修改为您的实际路径
    
    try:
        stats, dataset_structure = analyze_dataset(dataroot)
        results = print_statistics(stats)
        
        # 保存结果为CSV
        df = pd.DataFrame(results)
        df.to_csv('dataset_statistics.csv', index=False)
        print(f"\n统计结果已保存到 dataset_statistics.csv")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")