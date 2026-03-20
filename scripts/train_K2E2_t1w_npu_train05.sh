#!/bin/bash
#SBATCH --job-name=K2E_train05
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

# test minimum data used in fastcut mode
torchrun --nproc_per_node=8 train.py \
  --dataroot /public/home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T_train05 \
  --name T1w_K2E2_npu_resnet_6blocks_patchnce_train05 \
  --netG resnet_6blocks \
  --netD multiscale \
  --num_D 3 \
  --nce_layers 0,4,8,12,17 \
  --model cut \
  --CUT_mode FastCUT \
  --direction AtoB \
  --no_flip \
  --n_layers_D 3 \
  --save_epoch_freq 10 \
  --dataset_mode monai \
  --batch_size 8 \
  --input_nc 1 \
  --output_nc 1 \
  --ngf 64 \
  --gan_mode lsgan \
  --num_threads 8 \
  --n_epochs 50 \
  --n_epochs_decay 50 \
  --lr 0.0002 \
  --preprocess scale_width_and_crop \
  --load_size 256 \
  --display_id 0 \
  --lambda_SSIM 5 \
  --lambda_canny 5 \
  --lambda_elastic 5 \
  --lambda_perceptual 3.0 \
  --pixel_dim 1.0 1.0 -1 

