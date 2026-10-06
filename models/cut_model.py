import os
import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
# from .ranknce import  RankNCELoss as PatchNCELoss
import util.util as util
from pytorch_msssim import ssim, SSIM
import torch.nn.functional as F
import kornia

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        
        parser.add_argument('--lambda_SSIM', type=float, default=1, help='weight for SSIM loss')
        parser.add_argument('--lambda_canny', type=float, default=1, help='weight for Canny edge consistency loss')
        parser.add_argument('--lambda_elastic', type=float, default=1, help='weight for adaptive elastic loss')
        parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='weight for VGG perceptual loss')
        
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'D', 'NCE', 'Idt', 'SSIM', 'Canny', 'Elastic', 'Perceptual']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
        if self.isTrain and opt.lambda_perceptual > 0:
            self._init_perceptual_loss()
        # self.ssim_loss = SSIM(data_range=2.0, size_average=True, channel=opt.output_nc)
    def evaluate_quality(self, src, tgt):
        """综合评估生成质量 - 优化版，减少GPU-CPU同步"""
        with torch.no_grad():
            src_01 = (src + 1) / 2
            tgt_01 = (tgt + 1) / 2

            # 1. 结构相似度（越高越好）
            from pytorch_msssim import ssim
            ssim_score = ssim(src_01, tgt_01, data_range=1.0, size_average=True)

            # 2. 边缘保持率（越高越好）
            brain_mask = (src_01 > 0.1).float()
            kernel = torch.ones(1, 1, 9, 9, device=brain_mask.device) / 81.0
            eroded = F.conv2d(brain_mask, kernel, padding=4)
            eroded = (eroded > 0.98).float()
            edge_mask = brain_mask - eroded

            edge_mask_sum = edge_mask.sum()
            if edge_mask_sum > 0:
                edge_lost = ((tgt_01 * edge_mask) < 0.15).sum()
                edge_preserve_rate = 1 - (edge_lost / edge_mask_sum)
            else:
                edge_preserve_rate = torch.tensor(1.0, device=src.device)

            # 3. 清晰度（高频能量，越高越清晰）
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=src.dtype, device=src.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=src.dtype, device=src.device).view(1, 1, 3, 3)

            grad_tgt = torch.sqrt(
                F.conv2d(tgt_01, sobel_x, padding=1)**2 +
                F.conv2d(tgt_01, sobel_y, padding=1)**2 + 1e-8
            )
            sharpness = grad_tgt.mean()

            # 4. 综合分数（全部在GPU上计算）
            quality_score = (
                0.4 * ssim_score +          # 结构相似度40%
                0.4 * edge_preserve_rate +  # 边缘保持40%
                0.2 * torch.clamp(sharpness / 0.2, 0.0, 1.0)  # 清晰度20%（归一化，使用clamp替代min）
            )

            # 只在最后一步转换为float（调用者需要时再同步）
            return {
                'ssim': ssim_score,
                'edge_preserve': edge_preserve_rate,
                'sharpness': sharpness,
                'quality_score': quality_score
            }


    def _init_perceptual_loss(self):
        """
        初始化 VGG19 用于感知loss
        """
        from torchvision import models
        
        print("Initializing VGG19 for perceptual loss...")
        
        # 加载预训练的 VGG19
        vgg19 = models.vgg19(pretrained=True)
        
        # 提取特征层（到 relu5_4，即第 36 层）
        self.vgg = vgg19.features[:36].eval()
        
        # 移动到设备
        self.vgg.to(self.device)
        
        # 冻结参数
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # 选择用于计算loss的层
        # relu1_2(3), relu2_2(8), relu3_4(17), relu4_4(26), relu5_4(35)
        self.perceptual_layers = [3, 8, 17, 26, 35]
        self.perceptual_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  # 深层权重更大
        
        print("✅ VGG19 perceptual loss initialized successfully")
        print(f"   Using layers: {self.perceptual_layers}")
            
    
    def compute_perceptual_loss(self, src, tgt):
        """
        计算 VGG 感知 loss
        
        这个loss帮助保持图像的纹理和高层语义特征
        对��减少模糊非常有效
        """
        if not hasattr(self, 'vgg') or self.vgg is None:
            return torch.tensor(0.0, device=src.device)
        
        # 转换到 [0, 1]
        src_01 = (src + 1) / 2
        tgt_01 = (tgt + 1) / 2
        
        # 如果是单通道，复制为3通道（VGG需要3通道输入）
        if src_01.shape[1] == 1:
            src_01 = src_01.repeat(1, 3, 1, 1)
            tgt_01 = tgt_01.repeat(1, 3, 1, 1)
        
        # ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406], device=src.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=src.device).view(1, 3, 1, 1)
        
        src_norm = (src_01 - mean) / std
        tgt_norm = (tgt_01 - mean) / std
        
        # 提取多层特征并计算loss
        loss = 0.0
        src_feat = src_norm
        tgt_feat = tgt_norm
        
        current_layer = 0
        for i, layer in enumerate(self.vgg):
            src_feat = layer(src_feat)
            tgt_feat = layer(tgt_feat)
            
            # 如果当前层是我们要计算loss的层
            if i in self.perceptual_layers:
                layer_idx = self.perceptual_layers.index(i)
                weight = self.perceptual_weights[layer_idx]
                
                # L1 loss on features
                layer_loss = F.l1_loss(src_feat, tgt_feat)
                loss += weight * layer_loss
        
        return loss

    # ----------- SSIM -----------
    def compute_ssim_loss(self, src, tgt):
        # src, tgt: [-1, 1] -> 映射到 [0, 1] 再算 SSIM，更直观
        src = (src + 1) / 2
        tgt = (tgt + 1) / 2
        return 1 - ssim(src, tgt, data_range=1.0, size_average=True)

    # ----------- Sobel 梯度一致性 -----------
    def sobel_grad_loss(self, src, tgt):
        def sobel(x):
            # 简单 3×3 sobel 核
            kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            # 创建独立的 kernel_y 张量，而不是对 kernel_x 进行 transpose 操作
            kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            gx = F.conv2d(x, kernel_x, padding=1)
            gy = F.conv2d(x, kernel_y, padding=1)
            return gx, gy
        sx_x, sx_y = sobel(src)
        sy_x, sy_y = sobel(tgt)
        return F.l1_loss(sx_x, sy_x) + F.l1_loss(sx_y, sy_y)
    
    def compute_canny_loss(self, src, tgt, 
                           low_threshold=0.1, 
                           high_threshold=0.2, 
                           kernel_size=5, 
                           sigma=1.0, 
                           hysteresis=True, 
                           loss_type='l1', 
                           use_edge_map=False):
        """
        Args:
            low_threshold (float): Canny 低阈值。
            high_threshold (float): Canny 高阈值。
            kernel_size (tuple): 高斯核大小。
            sigma (tuple): 高斯核标准差。
            hysteresis (bool): 是否使用滞后阈值。
            loss_type (str): 损失类型，'l1' 或 'l2'。
            use_edge_map (bool): 如果为 True，则使用二值化边缘图计算损失；
                                 如果为 False，则使用梯度幅值图计算损失。
        """
        # 1. 转换到 [0, 1] 范围（kornia 期望的输入）
        src_01 = (src + 1) / 2
        tgt_01 = (tgt + 1) / 2
        
        # 2. 如果是多通道，转换为灰度
        if src_01.shape[1] == 3:
            src_gray = kornia.color.rgb_to_grayscale(src_01)
            tgt_gray = kornia.color.rgb_to_grayscale(tgt_01)
        else:
            src_gray = src_01
            tgt_gray = tgt_01
            
        magnitude_src, edges_src = kornia.filters.canny(
            src_gray,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            kernel_size=(kernel_size,kernel_size),
            sigma=(sigma, sigma),
            hysteresis=hysteresis
        )
        
        magnitude_tgt, edges_tgt = kornia.filters.canny(
            tgt_gray,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            kernel_size=(kernel_size,kernel_size),
            sigma=(sigma, sigma),
            hysteresis=hysteresis
        )
        
        # 4. 计算损失
        if use_edge_map:
            pred = edges_src
            target = edges_tgt
        else:
            pred = magnitude_src
            target = magnitude_tgt
        
        criterion = F.l1_loss if loss_type == 'l1' else F.mse_loss
        loss = criterion(pred, target)        
        
        return loss
    
    def compute_adaptive_elastic_loss(self, src, tgt, window_size=7):
        """
        无监督自适应弹性约束损失         
        核心思想：不依赖绝对强度值，而是依赖局部相对关系和变化幅度的一致性
        src: [-1,1],
        tgt: [-1,1]
        """
        src = (src + 1) / 2  # 映射到 [0, 1]
        tgt = (tgt + 1) / 2
        b, c, h, w = src.shape
        
        # 1. 计算局部统计特征（使用可分离卷积提高效率）
        pad = window_size // 2
        
        # 平均池化获取局部均值
        local_mean_src = F.avg_pool2d(src, window_size, stride=1, padding=pad, count_include_pad=False)
        local_mean_tgt = F.avg_pool2d(tgt, window_size, stride=1, padding=pad, count_include_pad=False)
        
        # 局部标准差（使用近似计算避免开方）
        local_sq_src = F.avg_pool2d(src**2, window_size, stride=1, padding=pad, count_include_pad=False)
        local_sq_tgt = F.avg_pool2d(tgt**2, window_size, stride=1, padding=pad, count_include_pad=False)
        local_std_src = torch.sqrt(torch.clamp(local_sq_src - local_mean_src**2, min=0.0) + 1e-8)
        
        # 2. 自适应弹性阈值（基于局部对比度动态调整）
        # 高对比度区域（如WM-GM边界）允许较大变化，低对比度区域（如均匀组织）严格约束
        local_contrast = local_std_src  # 用局部std作为对比度指标
        elasticity = torch.tanh(local_contrast * 5.0) * 0.2 + 0.1  # 范围 [0.1, 0.3]
        
        # 3. 计算相对变化（相对于局部均值的偏差变化）
        src_rel = src - local_mean_src  # 局部相对强度
        tgt_rel = tgt - local_mean_tgt
        diff_rel = tgt_rel - src_rel
        
        abs_diff = torch.abs(diff_rel)
        
        # Huber Loss with element-wise beta (elasticity)
        mask_elastic  = abs_diff <= elasticity
        mask_rigid  = ~mask_elastic
        loss_elastic = (0.5 * diff_rel ** 2 / (elasticity + 1e-8)) * mask_elastic.float()
        loss_rigid = (elasticity * (abs_diff - 0.5 * elasticity)) * mask_rigid.float()
        
        loss_total = (loss_elastic + loss_rigid).mean() # range [0, 0.5*elasticity]，平均后通常较小
        
        # 结构相关性
        corr_loss = self.local_correlation_loss(src, tgt, window_size)
        
        return loss_total + 0.2 * corr_loss
    
    def local_correlation_loss(self, src, tgt, window_size=7):
        """
        局部相关性损失：确保harmonization后局部结构关系保持
        """
        pad = window_size // 2
        src_patches = F.unfold(src, window_size, padding=pad) # [B, C*k*k, H*W]
        tgt_patches = F.unfold(tgt, window_size, padding=pad)
        
        # 使用标准化互相关（NCC）代替Spearman，更快且效果相当
        src_mean = src_patches.mean(dim=1, keepdim=True) # [B, 1, H*W]
        tgt_mean = tgt_patches.mean(dim=1, keepdim=True)
        
        src_std = torch.std(src_patches, dim=1, keepdim=True, unbiased=False)
        tgt_std = torch.std(tgt_patches, dim=1, keepdim=True, unbiased=False)
        
        src_std = torch.clamp(src_std, min=1e-6)
        tgt_std = torch.clamp(tgt_std, min=1e-6)
        
        # 计算协方差
        src_centered = src_patches - src_mean
        tgt_centered = tgt_patches - tgt_mean
        covariance = (src_centered * tgt_centered).mean(dim=1)  # [B, H*W]
        
        # NCC
        ncc = covariance / (src_std.squeeze(1) * tgt_std.squeeze(1))
        ncc = torch.clamp(ncc, -1.0, 1.0)
        ncc = torch.nan_to_num(ncc, nan=0.0)
        
        return ((1 - ncc) ** 2).mean() # range [0, 4]

    def data_dependent_initialize(self, data, accelerator=None):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        
        Parameters:
            data: input data sample
            accelerator: Accelerate Accelerator instance (optional)
        """
        self.set_input(data)
        
        with torch.no_grad():
            self.forward()
            if self.opt.isTrain:
                _ = self.compute_D_loss()
                _ = self.compute_G_loss()
        
        # Load checkpoint for continue training
        if self.opt.isTrain and self.opt.continue_train:
            print(f"\n{'='*70}")
            print("Continue training: Loading checkpoint after MLP initialization...")
            print(f"{'='*70}\n")
            
            load_suffix = self.opt.epoch
            
            # Load netF only (G and D already loaded in setup)
            load_filename = '%s_net_F.pth' % load_suffix
            load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name) if self.opt.pretrained_name else self.save_dir
            load_path = os.path.join(load_dir, load_filename)
            
            if os.path.exists(load_path):
                print(f'Loading netF from {load_path}')
                state_dict = torch.load(load_path, map_location=str(self.device))
                
                # Handle DDP/DataParallel wrappers
                net_f = self.netF
                while hasattr(net_f, 'module'):
                    net_f = net_f.module
                
                missing_keys, unexpected_keys = net_f.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f'  Warning: Missing keys: {missing_keys}')
                if unexpected_keys:
                    print(f'  Warning: Unexpected keys: {unexpected_keys}')
                
                print("✅ netF checkpoint loaded successfully")
            else:
                print(f"Warning: netF checkpoint not found at {load_path}")
        
        if self.opt.isTrain:
            if self.opt.lambda_NCE > 0.0 and any(p.requires_grad for p in self.netF.parameters()):
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
                
                if self.opt.continue_train:
                    self._load_optimizer_state(load_suffix)

    def _load_optimizer_state(self, epoch):
        """加载优化器状态（可选）"""
        try:
            opt_f_path = os.path.join(self.save_dir, f'{epoch}_optimizer_F.pth')
            if os.path.exists(opt_f_path):
                print(f'Loading optimizer_F state from {opt_f_path}')
                opt_state = torch.load(opt_f_path, map_location=str(self.device))
                self.optimizer_F.load_state_dict(opt_state)
                print("✅ optimizer_F state loaded")
        except Exception as e:
            print(f"Warning: Could not load optimizer_F state: {e}")

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        # with torch.autograd.detect_anomaly(False):
        self.backward(self.loss_D)
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if hasattr(self, 'optimizer_F'):
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        # with torch.autograd.detect_anomaly(False):
        self.backward(self.loss_G)
        self.optimizer_G.step()
        if hasattr(self, 'optimizer_F'):
            self.optimizer_F.step()
            
    def evaluate_epoch_end(self):
        """
        在每个epoch结束时调用，进行质量评估
        优化：减少GPU-CPU同步点
        """
        with torch.no_grad():
            # 使用一个batch进行评估（通常是最后一个batch）
            if hasattr(self, 'real_A') and hasattr(self, 'fake_B'):
                metrics = self.evaluate_quality(self.real_A, self.fake_B)
                
                # 批量转换为float（减少同步次数）
                ssim_val = float(metrics['ssim'])
                edge_val = float(metrics['edge_preserve'])
                sharp_val = float(metrics['sharpness'])
                score_val = float(metrics['quality_score'])

                print(f"\n{'='*60}")
                print(f"[Quality Metrics - Epoch {self.current_epoch}]")
                print(f"  SSIM:          {ssim_val:.4f}")
                print(f"  Edge Preserve: {edge_val:.4f}")
                print(f"  Sharpness:     {sharp_val:.4f}")
                print(f"  Quality Score: {score_val:.4f}")
                print(f"{'='*60}\n")
                
                # 保存历史记录（存储tensor，避免重复转换）
                if not hasattr(self, '_quality_history'):
                    self._quality_history = []
                
                # 存储标量值而非tensor，避免内存泄漏
                self._quality_history.append({
                    'epoch': self.current_epoch,
                    'quality_score': score_val,
                    'ssim': ssim_val,
                    'edge_preserve': edge_val,
                    'sharpness': sharp_val
                })
                
                # 检测最佳点
                if len(self._quality_history) >= 2:
                    previous_best = max([m['quality_score'] for m in self._quality_history[:-1]])
                    
                    if score_val > previous_best:
                        improvement = score_val - previous_best
                        print(f"  ✅ New best quality score! (Improved by {improvement:.4f})")
                        self.save_networks('best_quality')
                    
                    # 检测过拟合趋势
                    if len(self._quality_history) >= 4:
                        recent_scores = [m['quality_score'] for m in self._quality_history[-4:]]
                        if all(recent_scores[i] > recent_scores[i+1] for i in range(3)):
                            print(f"  ⚠️  Warning: Quality declining for 3 consecutive epochs!")
                            print(f"  Consider early stopping or using checkpoint from epoch {self._quality_history[-4]['epoch']}")

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths'] 
        # range [-1, 1]
        # print(f"real_A range: {self.real_A.min().item()} ~ {self.real_A.max().item()}")
        # print(f"real_B range: {self.real_B.min().item()} ~ {self.real_B.max().item()}")

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        
        # 前向传播
        pred_fake = self.netD(fake)
        pred_real = self.netD(self.real_B)
        
        # ⭐ 统一处理：如果不是 list，转换为 list
        if not isinstance(pred_fake, list):
            pred_fake = [pred_fake]
            pred_real = [pred_real]
        
        # 对所有尺度计算 loss（单尺度时只有 1 个元素）
        self.loss_D_fake = 0.0
        self.loss_D_real = 0.0
        
        for pred_fake_i, pred_real_i in zip(pred_fake, pred_real):
            self.loss_D_fake += self.criterionGAN(pred_fake_i, False).mean()
            self.loss_D_real += self.criterionGAN(pred_real_i, True).mean()
        
        # 平均
        num_scales = len(pred_fake)
        self.loss_D_fake = self.loss_D_fake / num_scales
        self.loss_D_real = self.loss_D_real / num_scales
        
        # 总 loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            
            # 统一处理：如果不是 list，转换为 list
            if not isinstance(pred_fake, list):
                pred_fake = [pred_fake]
            
            # 对所有尺度计算 loss
            self.loss_G_GAN = 0.0
            for pred_fake_i in pred_fake:
                self.loss_G_GAN += self.criterionGAN(pred_fake_i, True).mean()
            
            # 平均并乘以权重
            num_scales = len(pred_fake)
            self.loss_G_GAN = (self.loss_G_GAN / num_scales) * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        self.loss_NCE_Y = 0.0
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        
        self.loss_SSIM = 0.0
        self.loss_Canny = 0.0
        if self.opt.lambda_SSIM > 0:
            self.loss_SSIM = self.compute_ssim_loss(self.real_A, self.fake_B) * self.opt.lambda_SSIM

        if self.opt.lambda_canny > 0:
            self.loss_Canny = self.compute_canny_loss(self.real_A, self.fake_B) * self.opt.lambda_canny
        
        self.loss_Idt = 0.0
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_Idt = self.criterionIdt(self.real_B, self.idt_B) # not used now
        
        
        self.loss_Elastic = 0.0
        if self.opt.lambda_elastic > 0:
            self.loss_Elastic = self.compute_adaptive_elastic_loss(self.real_A, self.fake_B) * self.opt.lambda_elastic
        
        self.loss_Perceptual = 0.0
        if self.opt.lambda_perceptual > 0:
            self.loss_Perceptual = self.compute_perceptual_loss(self.real_A, self.fake_B) * self.opt.lambda_perceptual
        # total generator loss
        self.loss_G = (self.loss_G_GAN + 
                   loss_NCE_both + 
                   self.loss_SSIM + 
                   self.loss_Canny + 
                   self.loss_Elastic +
                     self.loss_Perceptual)
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None, src)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids, src)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
