import os
from pathlib import Path

def convert_t1w_to_t2w_split(t1w_split_dir):
    """
    根据T1w划分结果生成对应的T2w划分
    
    Args:
        t1w_split_dir: T1w划分结果的目录路径
    """
    t1w_split_dir = Path(t1w_split_dir)
    
    # 构建对应的T2w目录路径
    t2w_split_dir = t1w_split_dir.parent.parent / 'T2w' / t1w_split_dir.name
    t2w_split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"T1w划分目录: {t1w_split_dir}")
    print(f"T2w划分目录: {t2w_split_dir}")
    
    # 定义T1w和T2w数据根目录
    t1w_data_root = Path('/public/home_data/home/songhy2024/data/PVWMI/T1w/skullstripped/')
    t2w_data_root = Path('/public/home_data/home/songhy2024/data/PVWMI/T2w/skullstripped/')
    
    # 遍历T1w划分目录中的所有子目录（trainA, valA, testA, trainB, valB, testB）
    for split_name in ['trainA', 'valA', 'testA', 'trainB', 'valB', 'testB']:
        t1w_split_subdir = t1w_split_dir / split_name
        
        if not t1w_split_subdir.exists():
            print(f"跳过不存在的目录: {t1w_split_subdir}")
            continue
            
        print(f"\n处理划分: {split_name}")
        print(f"T1w源目录: {t1w_split_subdir}")
        
        # 创建对应的T2w子目录
        t2w_split_subdir = t2w_split_dir / split_name
        t2w_split_subdir.mkdir(exist_ok=True)
        print(f"T2w目标目录: {t2w_split_subdir}")
        
        successful_links = 0
        missing_files = []
        
        # 遍历T1w子目录中的所有符号链接
        for t1w_link_path in t1w_split_subdir.iterdir():
            if not t1w_link_path.is_symlink():
                print(f"跳过非符号链接: {t1w_link_path}")
                continue
            
            try:
                # 获取T1w符号链接指向的实际文件路径
                t1w_target_path = t1w_link_path.resolve()
                
                # 将T1w路径转换为对应的T2w路径
                # 方法：获取相对于T1w数据根目录的相对路径，然后应用到T2w数据根目录
                try:
                    relative_path = t1w_target_path.relative_to(t1w_data_root)
                except ValueError:
                    # 如果路径不在预期的根目录下，尝试直接转换文件名
                    print(f"警告: {t1w_target_path} 不在预期根目录下")
                    relative_path = Path(t1w_target_path.name)
                
                # 构建T2w文件名（将t1替换为t2）
                t2w_filename = relative_path.name.replace('_t1', '_t2')
                t2w_relative_path = relative_path.parent / t2w_filename
                
                # 构建完整的T2w文件路径
                t2w_target_path = t2w_data_root / t2w_relative_path
                
                # 检查T2w文件是否存在
                if not t2w_target_path.exists():
                    missing_files.append(str(t2w_target_path))
                    continue
                
                # 在T2w目录中创建符号链接
                t2w_link_path = t2w_split_subdir / t2w_target_path.name
                
                # 删除已存在的符号链接
                if t2w_link_path.exists() or t2w_link_path.is_symlink():
                    if t2w_link_path.is_symlink():
                        t2w_link_path.unlink()
                    else:
                        print(f"警告: {t2w_link_path} 已存在但不是符号链接，跳过")
                        continue
                
                # 创建相对路径符号链接[6,7](@ref)
                relative_source = os.path.relpath(t2w_target_path, start=t2w_split_subdir)
                t2w_link_path.symlink_to(relative_source)
                successful_links += 1
                print(f"创建符号链接: {t2w_link_path.name} -> {relative_source}")
                
            except OSError as e:
                print(f"处理文件 {t1w_link_path} 时出错: {e}")
            except Exception as e:
                print(f"处理文件 {t1w_link_path} 时出现意外错误: {e}")
        
        print(f"成功创建T2w符号链接: {successful_links}")
        
        if missing_files:
            print(f"缺失的T2w文件 ({len(missing_files)}个):")
            for missing_file in missing_files[:5]:
                print(f"  - {missing_file}")
            if len(missing_files) > 5:
                print(f"  ... 还有 {len(missing_files) - 5} 个文件")
    
    return t2w_split_dir

def verify_t2w_split(t2w_split_dir):
    """验证T2w划分结果"""
    
    print("\n" + "="*50)
    print("验证T2w划分结果")
    print("="*50)
    
    total_expected = 0
    total_created = 0
    valid_links = 0
    
    # 首先统计T1w划分中的符号链接数量作为预期值
    t1w_split_dir = t2w_split_dir.parent.parent / 'T1w' / t2w_split_dir.name
    for split_name in ['trainA', 'valA', 'testA', 'trainB', 'valB', 'testB']:
        t1w_subdir = t1w_split_dir / split_name
        if t1w_subdir.exists():
            t1w_links = [f for f in t1w_subdir.iterdir() if f.is_symlink()]
            total_expected += len(t1w_links)
    
    # 统计T2w划分中的符号链接
    for split_name in ['trainA', 'valA', 'testA', 'trainB', 'valB', 'testB']:
        t2w_subdir = t2w_split_dir / split_name
        print(f"\n{split_name}:")
        
        if t2w_subdir.exists():
            t2w_links = list(t2w_subdir.iterdir())
            actual_files = len(t2w_links)
            print(f"  T2w符号链接数: {actual_files}")
            
            # 检查符号链接是否有效
            valid_count = 0
            for link_file in t2w_links:
                if link_file.is_symlink() and link_file.exists():
                    valid_count += 1
            
            print(f"  有效符号链接: {valid_count}/{actual_files}")
            valid_links += valid_count
            total_created += actual_files
        else:
            print(f"  目录不存在: {t2w_subdir}")
    
    print(f"\n总体统计:")
    print(f"总预期符号链接（基于T1w）: {total_expected}")
    print(f"总创建符号链接: {total_created}")
    print(f"总有效符号链接: {valid_links}")
    
    return total_created == total_expected

def main():
    # 设置T1w划分目录路径
    t1w_split_dir = Path('/public/home_data/home/songhy2024/data/PVWMI/T1w/k2I-SIEMENS-SKYRA-3.0T/')
    
    try:
        # 转换T1w划分到T2w划分
        t2w_split_dir = convert_t1w_to_t2w_split(t1w_split_dir)
        
        # 验证结果
        verification_passed = verify_t2w_split(t2w_split_dir)
        
        if verification_passed:
            print("\n✅ T2w划分创建成功！")
        else:
            print("\n⚠️ T2w划分创建完成，但存在一些文件缺失或符号链接问题")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
    
'''
python make_t2w_dataset.py 
'''