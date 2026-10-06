import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.current_epoch = opt.epoch_count

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            if not hasattr(self, 'schedulers'):  # ADD THIS CHECK
                self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            # Dynamic feature MLPs are restored after their first forward pass.
            names = [n for n in self.model_names if n != 'F'] if self.isTrain else self.model_names
            self.load_networks(opt.epoch, names=names)

        self.print_networks(opt.verbose)

    def prepare_training(self, accelerator):
        """Prepare concrete networks, after data-dependent netF initialization.

        BaseModel is a controller, not an nn.Module. Passing this object itself
        to Accelerator.prepare() leaves G/D/F/E unsynchronized.
        The loader already uses a DistributedSampler; do not shard it twice.
        """
        self.accelerator = accelerator
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            if any(p.requires_grad for p in net.parameters()):
                setattr(self, 'net' + name, accelerator.prepare(net))
        prepared = []
        for optimizer in self.optimizers:
            wrapped = accelerator.prepare(optimizer)
            for name, value in list(vars(self).items()):
                if name.startswith('optimizer_') and value is optimizer:
                    setattr(self, name, wrapped)
            prepared.append(wrapped)
        self.optimizers = prepared
        # Include netF, whose optimizer did not exist at setup() time.
        self.schedulers = [networks.get_scheduler(o, self.opt) for o in self.optimizers]

    def backward(self, loss):
        if hasattr(self, 'accelerator'):
            self.accelerator.backward(loss)
        else:
            loss.backward()

    def parallelize(self):
        """Move networks to the correct device.
        
        Note: Distributed training is handled by Accelerate's prepare() method.
        This method only handles device placement for single-GPU or CPU training.
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if net is not None:
                    net.to(self.device)

    def data_dependent_initialize(self, data, accelerator=None):
        """Data-dependent initialization (e.g., for CUT model's netF MLP creation)"""
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self, to_cpu=True):
        """Return training losses/errors.

        Parameters:
            to_cpu: If True, convert tensor to float (triggers sync);
                   If False, return tensor (no sync, good for frequent calls)
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                val = getattr(self, 'loss_' + name)
                if to_cpu:
                    errors_ret[name] = float(val)
                else:
                    errors_ret[name] = val
        return errors_ret

    def save_networks(self, epoch, accelerator=None):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
            accelerator -- Accelerate Accelerator instance (only saves on main process)
        """
        # Only save on main process when using Accelerate
        if accelerator is not None and not accelerator.is_main_process:
            return

        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                # Unwrap from DDP/DataParallel/Accelerate wrappers
                net_to_save = net
                while hasattr(net_to_save, 'module'):
                    net_to_save = net_to_save.module

                # Save state_dict to CPU without moving the original network
                state_dict_cpu = {k: v.cpu().clone() for k, v in net_to_save.state_dict().items()}
                torch.save(state_dict_cpu, save_path)

        # Synchronization belongs to the training loop: saving may be main-rank-only.

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch, names=None):
        """Load all the networks from the disk."""
        for name in (self.model_names if names is None else names):
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if self.opt.isTrain and self.opt.pretrained_name is not None:
                    load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
                else:
                    load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net' + name)
                
                # Handle DDP/DataParallel wrapper
                if isinstance(net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                    net = net.module
                
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # Use strict=False to allow partial loading (for dynamically created networks)
                missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
                
                # Print warning messages
                if missing_keys:
                    print('  Warning: Missing keys in net%s: %s...' % (name, missing_keys[:5]) if len(missing_keys) > 5 else '  Warning: Missing keys in net%s: %s' % (name, missing_keys))
                if unexpected_keys:
                    print('  Warning: Unexpected keys in net%s: %s...' % (name, unexpected_keys[:5]) if len(unexpected_keys) > 5 else '  Warning: Unexpected keys in net%s: %s' % (name, unexpected_keys))

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        return {}
