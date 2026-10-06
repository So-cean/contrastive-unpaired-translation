"""This package includes all the modules related to data loading and preprocessing
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import torch
import random
import numpy as np
from copy import copy


def _worker_init_fn(worker_id, rank=0):
    """Initialize random seeds for each DataLoader worker."""
    try:
        base_seed = torch.initial_seed() % (2**32)
    except Exception:
        base_seed = int(torch.empty((), dtype=torch.int64).random_().item()) % (2**32)
    seed = (base_seed + rank + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class _WorkerInitFn:
    """可序列化的 worker_init_fn 包装器"""
    def __init__(self, rank, seed_offset=0):
        self.rank = rank
        self.seed_offset = seed_offset

    def __call__(self, worker_id):
        _worker_init_fn(worker_id, self.rank + self.seed_offset)


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py"."""
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(f"In {dataset_filename}.py, there should be a subclass of BaseDataset with class name that matches {target_dataset_name} in lowercase.")

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, accelerator=None, seed_offset=0):
    """Create a dataset given the option.

    Parameters:
        opt: options
        accelerator: Accelerate Accelerator instance (optional, for DDP)
        seed_offset: additional seed offset for independent sampling (used in SB model)
    """
    data_loader = CustomDatasetDataLoader(opt, accelerator=accelerator, seed_offset=seed_offset)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, accelerator=None, seed_offset=0):
        """Initialize this class

        Args:
            opt: options
            accelerator: Accelerate Accelerator instance (optional, for DDP)
            seed_offset: additional seed offset for independent sampling (used in SB model dual dataloaders)
        """
        self.opt = opt
        self.accelerator = accelerator
        self.seed_offset = seed_offset

        dataset_class = find_dataset_using_name(opt.dataset_mode)
        dataset_opt = copy(opt)
        dataset_opt.seed_offset = seed_offset
        self.dataset = dataset_class(dataset_opt)
        generator = torch.Generator().manual_seed(42 + seed_offset)
        print(f"dataset [{type(self.dataset).__name__}] was created")

        # Create sampler if using Accelerate with DDP
        self.sampler = None
        if accelerator is not None and accelerator.num_processes > 1:
            # Use different seed for DistributedSampler to ensure independent shuffling

            self.sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=(not opt.serial_batches),
                drop_last=True,
                seed=42 + seed_offset
            )
            print(f"[DataLoader] Using DistributedSampler for rank {accelerator.process_index}/{accelerator.num_processes}, seed_offset={seed_offset}")
            
        # 设置 worker_init_fn（使用可序列化的类代替lambda）
        if int(opt.num_threads) > 0:
            worker_rank = accelerator.process_index if accelerator else 0
            worker_init = _WorkerInitFn(worker_rank, seed_offset=seed_offset)
        else:
            worker_init = None

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=(self.sampler is None and not opt.serial_batches),
            sampler=self.sampler,
            num_workers=int(opt.num_threads),
            pin_memory=True,
            # Re-spawn workers so dataset.current_epoch reaches worker copies.
            persistent_workers=False,
            generator=generator,
            drop_last=bool(opt.isTrain),
            worker_init_fn=worker_init,
            prefetch_factor=2 if int(opt.num_threads) > 0 else None,
            multiprocessing_context='spawn' if opt.num_threads > 0 else None,
        )

    def set_epoch(self, epoch):
        """Set epoch for dataset and sampler (for distributed training shuffle)."""
        self.dataset.current_epoch = epoch
        self.dataloader.generator.manual_seed(42 + self.seed_offset + epoch * 1000)
        if self.sampler is not None:
            if hasattr(self.dataset, 'set_epoch') and callable(getattr(self.dataset, 'set_epoch')):
                self.dataset.set_epoch(epoch)
            if hasattr(self.sampler, 'set_epoch'):
                # Use different epoch seed for independent shuffling in dual dataloaders
                self.sampler.set_epoch(epoch + self.seed_offset * 1000)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset."""
        if self.sampler is not None:
            return len(self.sampler)
        else:
            return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data."""
        for i, data in enumerate(self.dataloader):
            # Only check max_dataset_size in non-distributed mode
            if self.sampler is None and i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
