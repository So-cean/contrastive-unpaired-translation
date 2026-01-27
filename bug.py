import torch, torch.multiprocessing as mp, os, time
import torch_npu
from torch_npu.contrib import transfer_to_npu

def worker(rank):
    torch.npu.set_device(rank)
    # 模拟 DDP 里拿到的 tensor：slice + view，stride 不保证连续
    mask = torch.ones(1, 256, 256, device=f'npu:{rank}') * (rank % 2)   # 全 0 或全 1
    mask_bin = (mask.squeeze(0) > 0.5).float()
    mask_flat = mask_bin.view(-1)              # 可能不连续
    mb = mask_flat                             # 和训练代码一致
    weights = torch.ones_like(mb)
    background_mask = (mb <= 0.5)
    print(f'rank{rank} nonzero cnt={background_mask.sum()}')
    weights[background_mask] = 0.01            # 这一行会调 NonZero
    weights /= weights.sum()

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(worker, nprocs=8, args=())