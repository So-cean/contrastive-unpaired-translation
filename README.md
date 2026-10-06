# CUT for Medical Imaging

**基于 CUT 的医学影像非配对域转换实验仓库**：将 3D NIfTI 体数据转换为 2D 切片训练，逐切片推理后重建 NIfTI 体数据，面向 MRI 跨域 harmonization / image translation 研究。

Maintained by [So-cean](https://github.com/So-cean). Forked from [taesungp/contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation). CUT / FastCUT 方法来自原作者；本 fork 的工作是医学数据适配、实验网络与损失扩展、训练工程和体数据推理。

## 本 fork 做了什么

| 模块 | 实现 | 当前边界 |
| --- | --- | --- |
| 医学数据 | `data/monai_dataset.py`：NIfTI、RAS 方向、重采样、裁剪/填充、强度归一化、非配对切片采样 | 主流程为单通道 **2D**；读取 3D 文件不等于 3D 模型 |
| 网络实验 | ResNet、ConvNeXt / ConvNeXtV2、MONAI U-Net 变体、多尺度判别器 | 默认示例采用 ResNet + basic；其他组合需分别验证 NCE 特征层 |
| 损失实验 | PatchNCE，以及 SSIM、Canny、adaptive elastic、VGG19 perceptual | 可配置加权；尚未提供统一消融与性能结论 |
| 训练工程 | Accelerate 网络/优化器准备、分布式采样、动态 netF、权重加载与保存 | 支持单进程和分布式训练；完整多卡训练与混合精度仍待验证 |
| 医学影像推理 | `inference_monai.py` / `predict_monai.py`：逐切片生成并重建 `.nii.gz` | 输出位于**预处理后的空间**，并非原始采集网格 |
| SB / UNSB 探索 | `train_sb.py`、`models/sb_model.py`、条件网络、双数据流 | 实验分支；未完成端到端结果验证 |
| 3D 数据探索 | `data/monai3d_dataset.py` | 仅体数据入口；尚未接通经过验证的 3D G/D/NCE 训练链路 |

## 安装

从仓库根目录执行。建议独立的 Python 3.11 环境：

```bash
git clone https://github.com/So-cean/contrastive-unpaired-translation.git
cd contrastive-unpaired-translation
conda env create -f environment.yml
conda activate medical-cut
```

先按所用硬件安装匹配的 PyTorch 与 torchvision，再安装项目依赖：

```bash
# CPU 示例；CUDA / Ascend 环境请换成对应硬件的软件组合。
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Ascend 还需要匹配的驱动、CANN、`torch_npu`。本仓库不自动安装或升级这些系统组件。依赖文件给出兼容下限，不是完整锁定文件。

## 数据组织与预处理

```text
DATA_ROOT/
├── trainA/*.nii.gz
├── trainB/*.nii.gz
├── valA/*.nii.gz       # 可选，单独运行推理
├── valB/*.nii.gz
├── testA/*.nii.gz
└── testB/*.nii.gz
```

- A / B 表示源域与目标域；训练不要求逐病例配对。当前读取顶层 `.nii.gz`，不递归扫描，也不读取 `.nii`。
- 请先按受试者划分 train / val / test，再提取切片；训练脚本不会自动划分数据，也不自动运行验证集。
- 预处理顺序为 RAS → `--pixel_dim` 重采样 → 中心裁剪/填充至 `256 × 256` → 0.5–99.5 百分位归一化。默认 `1 1 -1` 保留 z 轴原间距，模型输入映射到 `[-1, 1]`。
- 当前 MONAI 数据入口的平面尺寸固定为 256；`--crop_size` / `--load_size` 不改变此尺寸。默认将全部预处理体数据缓存到内存，每个分布式进程都有自己的缓存。
- 训练保留非零像素数大于 1000 的切片；没有满足条件的切片时使用中间切片。A 遍历、B 按 epoch/index/seed_offset 随机采样。epoch 长度为两域有效切片数的较大值。

## 从一个可复查的 CUT 配置开始

```bash
DATA_ROOT=/path/to/data EXPERIMENT_NAME=medical_cut \
  bash scripts/train_medical.sh --n_epochs 100 --n_epochs_decay 100
```

该脚本只启动一个训练任务，默认单通道、ResNet-9、basic 判别器、batch size 1，不启动 Visdom。脚本显式关闭本 fork 新增的四项损失，方便先检查 CUT 主路径；这不代表与原论文设置完全相同。

CPU 调试可设置 `ACCELERATE_USE_CPU=true` 并传入 `--gpu_ids -1`。批大小为**每进程**批大小；训练丢弃不完整 batch，数据过少时请减小 batch size / 进程数。

在基础配置跑通后，逐项启用实验损失，例如：

```bash
DATA_ROOT=/path/to/data EXPERIMENT_NAME=medical_cut_ssim \
  bash scripts/train_medical.sh --lambda_SSIM 1
```

直接运行 `train.py` 时，四项扩展损失的原有默认权重仍为 1。启用 `--lambda_perceptual` 会加载 torchvision 的 ImageNet VGG19 权重，首次可能需要下载；关闭它或进行推理时不加载 VGG。损失的效果需要数据上的消融验证，不预设有提升。`Idt` 当前仅记录，未作为独立项加入总损失。

多进程训练可使用 `torchrun --nproc_per_node=<卡数> train.py` 并附加完整训练参数；Ascend 环境需先配置匹配的 CANN 和 torch_npu。`scripts/train_*.sh` 中其他带数据集名称的文件是历史集群实验配置，可能包含固定路径、分区和环境名；使用前逐项调整。统一入口是 `scripts/train_medical.sh`。

## 权重保存与继续训练

权重写入 `checkpoints/<name>/<epoch>_net_{G,D,F}.pth`，保存频率由 `--save_epoch_freq` / `--save_latest_freq` 控制。对应训练参数保存在 `train_opt.txt`。

```bash
DATA_ROOT=/path/to/data EXPERIMENT_NAME=medical_cut \
  bash scripts/train_medical.sh --continue_train --epoch latest --epoch_count 101
```

这属于**加载网络权重后继续训练**：G/D 在 setup 时加载，动态 F 在首次特征初始化后加载。没有完整保存/恢复全部优化器、学习率调度器和随机数状态，因此不是精确断点续训。保持网络、通道数、NCE 层和预处理参数与原训练一致。

## 推理并导出 NIfTI

以下命令与上面的默认训练网络匹配；若训练时修改了 `ngf`、`netG`、归一化等配置，推理必须同步修改。

```bash
python predict_monai.py \
  --dataroot /path/to/data --name medical_cut --epoch latest \
  --model cut --dataset_mode monai --netG resnet_9blocks \
  --input_nc 1 --output_nc 1 --direction AtoB \
  --pixel_dim 1 1 -1 --phase test --results_dir ./results
```

`predict_monai.py` 自动统计源域体数据数；`--phase all` 依次处理 train / val / test。只运行 `inference_monai.py` 时注意默认 `--num_test 50` 的上限。结果保存在 `results/<name>/<phase>_<epoch>/`，包括 `*_fake_B.nii.gz`、预处理源图 `*_real_A.nii.gz`，以及存在时的目标参考图。

生成图映射到 `[0, 1]`，并乘以预处理源图 `> 0` 的掩膜；输出采用 MONAI 变换后的 affine，不会反变换回原始空间，也不恢复原始 MRI 强度量纲。此入口按单通道 CUT 生成器设计，不能直接用于 SB 的时间条件推理。

## 待完成事项

尚待补齐：真实数据结果与消融、公开可分享的样例/权重、完整多卡训练和混合精度验证、严格的训练状态恢复、SB 完整训练与推理验证、3D 网络适配。`results_agg.py` 是历史结果整理脚本，不代表仓库已经提供统一评估基准。当前没有发布可核验的准确率或加速比。

## 代码导航

```text
data/                 数据入口与预处理
models/               CUT、实验 SB、网络与损失
options/              命令行参数
scripts/              通用入口与历史实验配置
train.py              CUT 主训练入口
train_sb.py           实验 SB 双数据流入口
inference.py          通用图像推理入口
inference_monai.py     单 phase 逐切片推理与 NIfTI 重建
predict_monai.py      phase 遍历与源体数据计数
docs/                 数据集说明与上游文档
```

## 来源与引用

请保留并引用原始 [CUT / FastCUT](https://github.com/taesungp/contrastive-unpaired-translation) 工作：Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu, *Contrastive Learning for Unpaired Image-to-Image Translation*, ECCV 2020。原始介绍及 BibTeX 保存在 [上游 README](docs/UPSTREAM_README.md)。

SB / 条件网络探索参考 [UNSB](https://github.com/cyclomon/UNSB)，其来源与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本 fork 不将上游算法列为原创贡献。[LICENSE](LICENSE) 保留原 CUT 许可证。
