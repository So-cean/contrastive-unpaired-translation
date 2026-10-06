#!/bin/bash
#SBATCH --job-name=K2E2_unsb
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

accelerate launch --num_processes 8 \
    --num_cpu_threads_per_process 8 \
    --mixed_precision bf16 train_sb.py \
    --dataroot /public/home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
    --name T1w_K2E2_npu_resnet_9blocks_patchnce_unsb \
    --netG resnet_9blocks_cond \
    --netD basic_cond \
    --netE basic_cond \
    --nce_layers 0,4,8,12,16,20 \
    --model sb \
    --direction AtoB \
    --no_flip \
    --n_layers_D 3 \
    --save_epoch_freq 20 \
    --dataset_mode monai \
    --batch_size 8 \
    --input_nc 1 \
    --output_nc 1 \
    --ngf 64 \
    --gan_mode lsgan \
    --num_threads 8 \
    --n_epochs 100 \
    --n_epochs_decay 100 \
    --lr 0.0002 \
    --preprocess scale_width_and_crop \
    --load_size 256 \
    --display_id 0 \
    --pixel_dim 1.0 1.0 -1
