#!/bin/bash
#SBATCH --job-name=K2E2
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
    --dataroot /public/home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T_train05 \
    --name T1w_K2E2_npu_resnet_6blocks_patchnce_train05 \
    --netG resnet_6blocks \
    --results_dir /public/home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T_train05/K2E2_pred/ \
    --model cut \
    --CUT_mode FastCUT \
    --direction AtoB \
    --no_dropout \
    --no_flip \
    --dataset_mode monai \
    --batch_size 16 \
    --input_nc 1 \
    --output_nc 1 \
    --preprocess scale_width_and_crop \
    --ngf 64 \
    --phase all \
    --epoch 100