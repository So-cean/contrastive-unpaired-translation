#!/bin/bash
#SBATCH --job-name=pred_CUT
#SBATCH --output=./slurm_logs/out_%j.log
#SBATCH --error=./slurm_logs/err_%j.log
#SBATCH --time=5-00:00:00
#SBATCH --partition=bme_a10080g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1

conda activate kaolin
cd /public_bme2/bme-wangqian2/songhy2024/contrastive-unpaired-translation

python predict_monai.py --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
    --results_dir /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_pred/ \
    --name CUT_monai_K2E2 --model cut --direction AtoB --no_dropout --no_flip --dataset_mode monai \
    --batch_size 16 --input_nc 1 --output_nc 1 --preprocess scale_width_and_crop --ngf 96 --phase all

# python predict_monai.py --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
#     --name CUT_monai_K2E2 --model cut --direction AtoB --no_dropout --no_flip --dataset_mode monai \
#     --batch_size 16 --input_nc 1 --output_nc 1 --preprocess scale_width_and_crop --ngf 96 --phase all


# python predict_monai.py  --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2I-SIEMENS-SKYRA-3.0T/ \
#     --results_dir /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2I-SIEMENS-SKYRA-3.0T/K2I_pred/ \
#     --name CUT_monai_K2I --model cut --direction AtoB --no_dropout --no_flip --dataset_mode monai \
#     --batch_size 16 --input_nc 1 --output_nc 1 --preprocess scale_width_and_crop --ngf 96 --phase all

# python predict_monai.py --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2I-SIEMENS-SKYRA-3.0T/ \
#     --results_dir /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2I-SIEMENS-SKYRA-3.0T/K2I_pred/ \
#     --name CUT_monai_K2I --model cut --direction AtoB --no_dropout --no_flip --dataset_mode monai \
#     --batch_size 16 --input_nc 1 --output_nc 1 --preprocess scale_width_and_crop --ngf 96 --phase all

torchrun --nproc_per_node=1 predict_monai.py --dataroot /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
    --results_dir /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_npu_pred_min/ \
    --name CUT_monai_K2E2_npu_min --model cut --direction AtoB --no_dropout --no_flip --dataset_mode monai \
    --batch_size 16 --input_nc 1 --output_nc 1 --preprocess scale_width_and_crop --ngf 96 --phase all

torchrun --nproc_per_node=1 predict_monai.py --dataroot /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
    --results_dir /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_npu_pred_convnext2/ \
    --name CUT_monai_K2E2_npu_convnext2 --model cut --direction AtoB --no_dropout --no_flip --dataset_mode monai \
    --netG convnext_24blocks \
    --nce_layers 1,8,14,23,30,35 \
    --batch_size 16 --input_nc 1 --output_nc 1 --preprocess scale_width_and_crop --ngf 64 --phase all

torchrun --nproc_per_node=1 predict_monai.py --dataroot /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
    --results_dir /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_npu_pred_convnext2/ \
    --name CUT_monai_K2E2_npu_convnext2 --model cut --direction AtoB --no_dropout --no_flip --dataset_mode monai \
    --netG convnext_24blocks \
    --nce_layers 1,8,14,23,30,35 \
    --batch_size 16 --input_nc 1 --output_nc 1 --preprocess scale_width_and_crop --ngf 64 --phase testbug

torchrun --nproc_per_node=1 predict_monai.py --dataroot /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
    --results_dir /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_npu_pred_convnext3/ \
    --name CUT_monai_K2E2_npu_convnext3 --model cut --direction AtoB --no_dropout --no_flip --dataset_mode monai \
    --netG convnext_24blocks \
    --nce_layers 1,8,14,23,30,35 \
    --batch_size 16 --input_nc 1 --output_nc 1 --preprocess scale_width_and_crop --ngf 64 --phase all

torchrun --nproc_per_node=1 predict_monai.py --dataroot /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/ \
    --results_dir /home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_npu_pred2/ \
    --name CUT_monai_K2E2_npu --model cut --direction AtoB --no_dropout --no_flip --dataset_mode monai \
    --batch_size 16 --input_nc 1 --output_nc 1 --preprocess scale_width_and_crop --ngf 96 --phase all --epoch 10