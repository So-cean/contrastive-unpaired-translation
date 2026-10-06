import numpy as np
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import os


class SBModel(BaseModel):
    """
    Schrödinger Bridge Model for unpaired image-to-image translation.
    Based on UNSB: https://github.com/cyclomon/UNSB
    Adapted for Accelerate distributed training.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configure options specific for SB model"""
        parser.add_argument('--mode', type=str, default="sb", choices=['sb', 'SB'])
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss')
        parser.add_argument('--lambda_SB', type=float, default=0.1, help='weight for SB loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                           help='use NCE loss for identity mapping')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16',
                           help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                           type=util.str2bool, nargs='?', const=True, default=False,
                           help='include negatives from entire minibatch')
        parser.add_argument('--netF', type=str, default='mlp_sample',
                           choices=['sample', 'reshape', 'mlp_sample'],
                           help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance', type=util.str2bool, nargs='?', const=True, default=False,
                           help="Enforce flip-equivariance as additional regularization")

        # SB-specific parameters
        parser.add_argument('--tau', type=float, default=0.01, help='noise level for SB')
        parser.add_argument('--num_timesteps', type=int, default=4, help='number of timesteps for SB')
        parser.add_argument('--std', type=float, default=0.1, help='std for noise injection')

        parser.set_defaults(pool_size=0, netG='resnet_9blocks_cond', netD='basic_cond', netE='basic_cond')

        opt, _ = parser.parse_known_args()
        if opt.mode.lower() == "sb":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # Loss names
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'SB', 'E']
        self.visual_names = ['real_A', 'real_A_noisy', 'fake_B', 'real_B']

        if self.opt.phase == 'test':
            self.visual_names = ['real']
            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'E']
        else:
            self.model_names = ['G']

        # Define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, opt.no_antialias_up,
                                      self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias,
                                      self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                          opt.normD, opt.init_type, opt.init_gain,
                                          opt.no_antialias, self.gpu_ids, opt)
            # Energy network E takes 4x channels (concatenated input and output)
            self.netE = networks.define_D(opt.output_nc * 4, opt.ndf, opt.netE, opt.n_layers_D,
                                          opt.normD, opt.init_type, opt.init_gain,
                                          opt.no_antialias, self.gpu_ids, opt)

            # Loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            # Optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr,
                                                betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr,
                                                betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr,
                                                betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)

    def data_dependent_initialize(self, data, data2=None, accelerator=None):
        """
        Initialize netF MLP with first batch.
        SB model supports dual data input for independent domain A and B sampling.
        """
        # Pass both data to set_input (data2 is optional for backward compatibility)
        self.set_input(data, data2=data2)
        
        with torch.no_grad():
            self.forward()
            if self.opt.isTrain:
                _ = self.compute_G_loss()
                _ = self.compute_D_loss()
                _ = self.compute_E_loss()
        
        if self.opt.isTrain and self.opt.continue_train:
            load_suffix = self.opt.epoch
            load_filename = '%s_net_F.pth' % load_suffix
            load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name) if self.opt.pretrained_name else self.save_dir
            load_path = os.path.join(load_dir, load_filename)

            if os.path.exists(load_path):
                print(f'Loading netF from {load_path}')
                state_dict = torch.load(load_path, map_location=str(self.device))
                net_f = self.netF
                while hasattr(net_f, 'module'):
                    net_f = net_f.module
                missing_keys, unexpected_keys = net_f.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f'  Warning: Missing keys: {missing_keys}')
                if unexpected_keys:
                    print(f'  Warning: Unexpected keys: {unexpected_keys}')
                print("✅ netF checkpoint loaded successfully")
        
        if self.opt.isTrain:
            if self.opt.lambda_NCE > 0.0 and any(p.requires_grad for p in self.netF.parameters()):
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                    betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

            if self.opt.continue_train:
                self._load_optimizer_state(self.opt.epoch)

    def _load_optimizer_state(self, epoch):
        """Load optimizer state for continue training"""
        try:
            opt_f_path = os.path.join(self.save_dir, f'{epoch}_optimizer_F.pth')
            if os.path.exists(opt_f_path):
                print(f'Loading optimizer_F state from {opt_f_path}')
                opt_state = torch.load(opt_f_path, map_location=str(self.device))
                self.optimizer_F.load_state_dict(opt_state)
                print("✅ optimizer_F state loaded")
        except Exception as e:
            print(f"Warning: Could not load optimizer_F state: {e}")

    def set_input(self, data, data2=None):
        """
        Unpack input data from the dataloader.
        SB model uses dual data input for independent sampling:
        - real_A: anchor from domain A (dataset 1)
        - real_A2: negative from domain A (dataset 2, independent sample for SB contrastive learning)
        - real_B: target from domain B (dataset 2)

        Args:
            data: First data batch (contains A and B, but A is used as anchor)
            data2: Second data batch (contains A and B with different random seed), optional
                   If None, uses single dataset mode (backward compatibility)
        """
        AtoB = self.opt.direction == 'AtoB'
        
        # real_A: anchor from first dataset's domain A
        self.real_A = data['A' if AtoB else 'B'].to(self.device)
        self.image_paths = data['A_paths' if AtoB else 'B_paths']

        if data2 is not None:
            # Dual dataset mode: dataset2 uses DIFFERENT random seed
            # -> data2['A'] is an INDEPENDENT sample from domain A (for SB contrastive learning)
            # -> data2['B'] is the target from domain B
            self.real_A2 = data2['A' if AtoB else 'B'].to(self.device)  # Negative sample
            self.real_B = data2['B' if AtoB else 'A'].to(self.device)   # Target
        else:
            # Single dataset mode (backward compatibility): create real_A2 by adding noise
            self.real_B = data['B' if AtoB else 'A'].to(self.device)
            with torch.no_grad():
                noise = torch.randn_like(self.real_A) * 0.01
                self.real_A2 = self.real_A + noise

    def forward(self):
        """Schrödinger Bridge forward pass"""
        tau = self.opt.tau
        T = self.opt.num_timesteps

        # Create time schedule
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        times = torch.tensor(times).float().to(self.device)
        self.times = times

        bs = self.real_A.size(0)
        time_idx = (torch.randint(T, size=[1], device=self.device) *
                    torch.ones(size=[1], device=self.device)).long()
        self.time_idx = time_idx
        self.timestep = times[time_idx]
        
        # Create second sample for SB if not exists
        if not hasattr(self, 'real_A2'):
            with torch.no_grad():
                noise = torch.randn_like(self.real_A) * 0.01
                self.real_A2 = self.real_A + noise

        # Forward diffusion process
        with torch.no_grad():
            self.netG.eval()
            for t in range(self.time_idx.int().item() + 1):
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1, 1, 1, 1)
                    scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                else:
                    inter = None
                    scale = None
                
                # Sample 1
                Xt = self.real_A if (t == 0) else ((1-inter) * Xt + inter * Xt_1.detach() +
                                                    (scale * tau).sqrt() * torch.randn_like(Xt))
                time_idx_t = (t * torch.ones(size=[bs], device=self.device)).long()
                time_t = times[time_idx_t]
                z = torch.randn(size=[bs, 4*self.opt.ngf], device=self.device)
                Xt_1 = self.netG(Xt, time_idx_t, z)
                
                # Sample 2 (for SB contrastive learning)
                Xt2 = self.real_A2 if (t == 0) else ((1-inter) * Xt2 + inter * Xt_12.detach() +
                                                     (scale * tau).sqrt() * torch.randn_like(Xt2))
                Xt_12 = self.netG(Xt2, time_idx_t, z)
                
                # Identity path
                if self.opt.nce_idt:
                    XtB = self.real_B if (t == 0) else ((1-inter) * XtB + inter * Xt_1B.detach() +
                                                       (scale * tau).sqrt() * torch.randn_like(XtB))
                    Xt_1B = self.netG(XtB, time_idx_t, z)

            if self.opt.nce_idt:
                self.XtB = XtB.detach()
            self.real_A_noisy = Xt.detach()
            self.real_A_noisy2 = Xt2.detach()
        
        # Training forward
        z_in = torch.randn(size=[2*bs, 4*self.opt.ngf], device=self.device)
        z_in2 = torch.randn(size=[bs, 4*self.opt.ngf], device=self.device)
        
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if (self.opt.nce_idt and self.opt.isTrain) else self.real_A
        self.realt = torch.cat((self.real_A_noisy, self.XtB), dim=0) if (self.opt.nce_idt and self.opt.isTrain) else self.real_A_noisy
        
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
                self.realt = torch.flip(self.realt, [3])
        
        self.fake = self.netG(self.realt, self.time_idx, z_in)
        self.fake_B2 = self.netG(self.real_A_noisy2, self.time_idx, z_in2)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        # Test mode: generate full trajectory
        if self.opt.phase == 'test':
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.opt.num_timesteps):
                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1, 1, 1, 1)
                        scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                    
                    Xt = self.real_A if (t == 0) else ((1-inter) * Xt + inter * Xt_1.detach() +
                                                        (scale * tau).sqrt() * torch.randn_like(Xt))
                    time_idx_t = (t * torch.ones(size=[self.real_A.shape[0]], device=self.device)).long()
                    z = torch.randn(size=[self.real_A.shape[0], 4*self.opt.ngf], device=self.device)
                    Xt_1 = self.netG(Xt, time_idx_t, z)
                    setattr(self, "fake_" + str(t+1), Xt_1)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        
        pred_fake = self.netD(fake, self.time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        pred_real = self.netD(self.real_B, self.time_idx)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()
        
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_E_loss(self):
        """Calculate Energy Network (E) loss for Schrödinger Bridge"""
        XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2.detach()], dim=1)
        
        # Compute energy function
        E_positive = self.netE(XtXt_1, self.time_idx, XtXt_1)
        E_negative = self.netE(XtXt_1, self.time_idx, XtXt_2)
        
        # Log-sum-exp for numerical stability
        temp = torch.logsumexp(E_negative.reshape(-1), dim=0)
        
        self.loss_E = -E_positive.mean() + temp + temp**2
        return self.loss_E

    def compute_G_loss(self):
        """Calculate GAN, NCE and SB loss for the generator"""
        tau = self.opt.tau
        fake = self.fake_B
        
        # GAN loss
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake, self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # SB (Schrödinger Bridge) loss
        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)
            
            ET_XY = (self.netE(XtXt_1, self.time_idx, XtXt_1).mean() -
                     torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0))

            self.loss_SB = -((self.opt.num_timesteps - self.time_idx[0]) / self.opt.num_timesteps *
                            self.opt.tau * ET_XY)
            self.loss_SB += self.opt.tau * torch.mean((self.real_A_noisy - self.fake_B)**2)

        # NCE loss
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE = 0.0

        self.loss_NCE_Y = 0.0
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        
        self.loss_G = self.loss_G_GAN + self.opt.lambda_SB * self.loss_SB + self.opt.lambda_NCE * loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        """Calculate contrastive loss"""
        n_layers = len(self.nce_layers)
        z = torch.randn(size=[self.real_A.size(0), 4*self.opt.ngf], device=self.real_A.device)

        feat_q = self.netG(tgt, self.time_idx * 0, z, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        
        feat_k = self.netG(src, self.time_idx * 0, z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def optimize_parameters(self):
        """Optimize networks"""
        # Forward
        self.forward()

        # Update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.backward(self.loss_D)
        self.optimizer_D.step()

        # Update E
        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.backward(self.loss_E)
        self.optimizer_E.step()

        # Update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)

        self.optimizer_G.zero_grad()
        if hasattr(self, 'optimizer_F'):
            self.optimizer_F.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.backward(self.loss_G)
        self.optimizer_G.step()
        if hasattr(self, 'optimizer_F'):
            self.optimizer_F.step()
