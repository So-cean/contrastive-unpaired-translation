#!/bin/bash
#SBATCH --job-name=K2I
#SBATCH --output=./slurm_logs/out_%j.log
#SBATCH --error=./slurm_logs/err_%j.log
#SBATCH --time=120:00:00
#SBATCH --partition=bme_npu        # NPU 分区
#SBATCH --nodes=1                  # 只用 1 个节点
#SBATCH --nodelist=bme-npu02
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=8         
#SBATCH --exclusive              

conda activate kaolin
cd /public/home_data/home/songhy2024/contrastive-unpaired-translation

torchrun --nproc_per_node=1 predict_monai.py \
    --dataroot /public/home_data/home/songhy2024/data/PVWMI/T2w/k2I-SIEMENS-SKYRA-3.0T/ \
    --name T2w_K2I_npu_resnet_9blocks_patchnce \
    --netG resnet_9blocks \
    --results_dir /public/home_data/home/songhy2024/data/PVWMI/T2w/k2I-SIEMENS-SKYRA-3.0T/K2I_pred/ \
    --model cut \
    --direction AtoB \
    --no_dropout \
    --no_flip \
    --dataset_mode monai \
    --batch_size 16 \
    --input_nc 1 \
    --output_nc 1 \
    --preprocess scale_width_and_crop \
    --ngf 64 \
    --phase all 