"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import torch
import random
import numpy as np
import torch.distributed as dist
import math


def _worker_init_fn(worker_id):
    """Initialize random seeds for each DataLoader worker.
    Uses torch.initial_seed() which is already different per worker; incorporate rank so different ranks don't share seeds.
    """
    try:
        base_seed = torch.initial_seed() % (2**32)
    except Exception:
        base_seed = int(torch.empty((), dtype=torch.int64).random_().item()) % (2**32)
    rank = 0
    try:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except Exception:
        rank = 0
    seed = (base_seed + rank + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)

        # Use DistributedSampler when running under torch.distributed
        self.sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset, 
                shuffle=(not opt.serial_batches),
                drop_last=True  # 确保所有 rank 的样本数一致
            )
            
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=(self.sampler is None and not opt.serial_batches),
            sampler=self.sampler,
            num_workers=int(opt.num_threads),
            pin_memory=True,
            persistent_workers=True if int(opt.num_threads) > 0 else False,
            drop_last=True,  # 始终 drop_last，确保所有 rank batch 数一致
            worker_init_fn=_worker_init_fn if int(opt.num_threads) > 0 else None,
            prefetch_factor=2,
            multiprocessing_context='spawn' if opt.num_threads > 0 else None,
        )
        # check sampler
        print(f"DataLoader created with sampler={self.dataloader.sampler}, type={type(self.dataloader.sampler)}")
        print(f"len sampler = {len(self.sampler) if self.sampler is not None else 'N/A'}")

    def set_epoch(self, epoch):
        self.dataset.current_epoch = epoch
        if self.sampler is not None:
            # 2. 调用 dataset 的 set_epoch 方法（新增：让 MonaiDataset 重新生成 A-B 配对）
            if hasattr(self.dataset, 'set_epoch') and callable(getattr(self.dataset, 'set_epoch')):
                self.dataset.set_epoch(epoch)
            # DistributedSampler needs to know the epoch for shuffling
            try:
                self.sampler.set_epoch(epoch)
            except Exception:
                pass

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset
        
        When running with DistributedSampler (DDP), return the number of samples assigned to this
        replica (so len(dataset) reflects the local partition). Otherwise return the dataset length
        truncated by opt.max_dataset_size as before.
        """
        if self.sampler is not None:
            # 注意：DistributedSampler 会自动分区数据，这里返回当前进程的样本数
            return len(self.sampler)
        else:
            return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            # 在 DDP 模式下不要截断，让所有 rank 迭代相同次数
            if self.sampler is None and i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
