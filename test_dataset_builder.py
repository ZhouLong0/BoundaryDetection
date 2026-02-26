import argparse
import os
import torch

from datasets import build_dataloader
from modeling import cfg

import torch.backends.cudnn as cudnn
import random, shutil
import numpy as np
import torch.distributed as dist
from utils.distribute import synchronize, all_gather, is_main_process


def init_seeds(seed, cuda_deterministic=True):
    cudnn.enabled = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def test_build_dataloader():
    """Test the build_dataloader function"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config-files/baseline.yaml')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--test-only", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expname", type=str, default='test')
    parser.add_argument("--all-thres", action='store_true', default=True, help='test using all thresholds [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.local_rank:
        args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1

    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = True
        init_seeds(args.seed + args.local_rank)
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", world_size=args.num_gpus, rank=args.local_rank)
            dist.barrier()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if is_main_process():
        print('Args: \n{}'.format(args))
        print('Configs: \n{}'.format(cfg))
    
    # Test build_dataloader
    train_data_loader = build_dataloader(cfg, args, cfg.DATASETS.TEST, is_train=False)
    
    print(f"Train dataloader created successfully!")
    print(f"Dataset size: {len(train_data_loader.dataset)}")
    print(f"Batch size: {train_data_loader.batch_size}")
    print(f"Number of batches: {len(train_data_loader)}")
    
    # Test one batch: print full values (tensor contents and non-tensor values)
    for batch in train_data_loader:
        print(f"\nBatch keys: {list(batch.keys())}")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                try:
                    v = val.cpu().numpy()
                    print(f"  {key} (tensor) shape={val.shape} dtype={val.dtype} values=\n{v}")
                except Exception:
                    # Fallback if conversion fails
                    print(f"  {key} (tensor) shape={val.shape} dtype={val.dtype} values=\n{val}")
            else:
                print(f"  {key} ({type(val)}) value=\n{val}")
        break


if __name__ == '__main__':
    test_build_dataloader()