import torch
import torch.distributed as dist
import os


def setup_distributed_gpus():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )

        dist.barrier(device_ids=[local_rank])
        return device, rank, world_size

    # single‑GPU/CPU fallback
    if torch.cuda.is_available():
        return torch.device("cuda"), 0, 1
    else:
        return torch.device("cpu"), 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
