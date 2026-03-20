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

torchrun --nproc_per_node=8 train.py \
  --dataroot /public/home_data/home/songhy2024/data/PVWMI/T2w/k2I-SIEMENS-SKYRA-3.0T/ \
  --name T2w_K2I_npu_resnet_9blocks_patchnce \
  --netG resnet_9blocks \
  --netD multiscale \
  --num_D 3 \
  --nce_layers 0,4,8,12,16,20 \
  --model cut \
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
  --n_epochs 100 \
  --n_epochs_decay 20 \
  --lr 0.0002 \
  --preprocess scale_width_and_crop \
  --load_size 256 \
  --display_id 0 \
  --lambda_SSIM 5 \
  --lambda_canny 5 \
  --lambda_elastic 5 \
  --lambda_perceptual 3.0 \
  --pixel_dim 1.0 1.0 -1  