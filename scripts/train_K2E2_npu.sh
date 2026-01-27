#!/bin/bash
#SBATCH --job-name=CUT
#SBATCH --output=./slurm_logs/out_%j.log
#SBATCH --error=./slurm_logs/err_%j.log
#SBATCH --time=5-00:00:00
#SBATCH --partition=bme_a10080g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1

conda activate kaolin
cd /public/home_data/home/songhy2024/contrastive-unpaired-translation

torchrun --nproc_per_node=8 train.py \
  --dataroot /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
  --name CUT_monai_K2E2_npu \
  --model cut \
  --direction AtoB \
  --no_dropout \
  --no_flip \
  --n_layers_D 3 \
  --save_epoch_freq 10 \
  --dataset_mode monai \
  --batch_size 12 \
  --input_nc 1 \
  --output_nc 1 \
  --ngf 96 \
  --gan_mode lsgan \
  --num_threads 8 \
  --n_epochs 50 \
  --n_epochs_decay 50 \
  --lr 0.0005 \
  --preprocess scale_width_and_crop \
  --load_size 256 \
  --display_id 0 \
  --pixel_dim 1.0 1.0 -1 


torchrun --nproc_per_node=1 train.py \
  --dataroot /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
  --name CUT_monai_K2E2_npu_single \
  --model cut \
  --direction AtoB \
  --no_dropout \
  --no_flip \
  --n_layers_D 3 \
  --save_epoch_freq 10 \
  --dataset_mode monai \
  --batch_size 12 \
  --input_nc 1 \
  --output_nc 1 \
  --ngf 96 \
  --gan_mode lsgan \
  --num_threads 8 \
  --n_epochs 50 \
  --n_epochs_decay 50 \
  --lr 0.0001 \
  --preprocess scale_width_and_crop \
  --load_size 256 \
  --display_id 0 \
  --pixel_dim 1.0 1.0 -1 

# python train.py --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ --name CUT_monai_E2K2 --model cut --direction BtoA \
#     --no_dropout --no_flip --n_layers_D 3 --save_epoch_freq 20 \
#     --dataset_mode monai --batch_size 16 --input_nc 1 --output_nc 1 --ngf 96 --gan_mode lsgan \
#     --num_threads 8 --n_epochs 50 --n_epochs_decay 50  --lr 0.0001 --preprocess scale_width_and_crop --load_size 256 \
#     --display_id 0
