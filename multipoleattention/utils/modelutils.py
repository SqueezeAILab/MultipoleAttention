import torch
import torch.nn as nn
import torch.distributed as dist
import os
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "2400"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
from datetime import timedelta

def init_dist(rank, world_size, port_num):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'  # or set to the appropriate master address
    os.environ['MASTER_PORT'] = str(port_num)    # ensure this port is free or choose another one

    torch.cuda.set_device(rank)

    timeout_long_ncll = timedelta(seconds=7200)  # 100 minutes
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size,
        timeout=timeout_long_ncll
    )
    global_group = dist.group.WORLD
    return rank, global_group

def shard_model(model, rank, world_size, tp_size, port_num=6060):
    global_rank, global_group = init_dist(rank, world_size, port_num)
    assert (global_rank == rank)

    def get_tp_rank():
        return dist.get_rank()

    def apply_tensor_parallelism(model, tp_size):
        namelist = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        colwise = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
        rowwise = ["o_proj", "down_proj"]
        for name, param in model.named_parameters():
            for n in namelist:
                if n in name and "weight" in name and param.dim() == 2:
                    if n in colwise:
                        shard_dim = 0
                    elif n in rowwise:
                        shard_dim = 1
                    else:
                        assert(False)
                    param.data = torch.chunk(param.data, tp_size, dim=shard_dim)[get_tp_rank()]
        return model


    model = apply_tensor_parallelism(model, tp_size)
    model.global_group = global_group
    return model
