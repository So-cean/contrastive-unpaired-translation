# 性能测量

先固定数据、网络、损失、batch size、精度和软件版本，再测量。`profile_performance.py` 用一个 batch 做预热与单进程训练步计时，会更新临时模型但不保存权重；它不代表端到端 epoch 吞吐，也不外推多卡速度。

```bash
python profile_performance.py --dataroot /path/to/data --dataset_mode monai \
  --model cut --input_nc 1 --output_nc 1 --num_threads 0 \
  --lambda_SSIM 0 --lambda_canny 0 --lambda_elastic 0 --lambda_perceptual 0
```

MONAI 当前完整缓存预处理体数据；先分别记录缓存构建时间、训练时间和内存占用。多进程会复制缓存。worker 每个 epoch 重新创建，以保证切片采样看到正确的 epoch；增大 num_threads 前先核对内存和进程预算。

依次比较 batch size、num_threads、多尺度判别器数量和各项附加损失。VGG 权重加载、异常检测、频繁打印与同步会影响计时；首次加载不可计入稳定训练步性能。多 NPU 必须实测全局 samples/s，不能用单卡速度直接乘卡数。

当前仓库不提供已经验证的加速比。记录测量结果时，应附上完整命令、硬件、版本、预热次数和统计区间。
