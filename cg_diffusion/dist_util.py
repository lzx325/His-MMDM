"""
Helpers for distributed training.
"""

import io
import os
import socket
import time

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(debug_log=False):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    comm = MPI.COMM_WORLD
    if debug_log:
        time.sleep(comm.rank*1)
        print(
            "setup_dist before everything",
            "cuda availability",th.cuda.is_available(),
            "CUDA_VISIBLE_DEVICES",os.environ.get("CUDA_VISIBLE_DEVICES",""),
            "torch.cuda.current_device()",th.cuda.current_device(),
            "torch.cuda.device_count()",th.cuda.device_count(),
            "GPUS_PER_NODE",GPUS_PER_NODE
        )

    device_id=MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE
    th.cuda.set_device(device_id)

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    
    
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)

    if debug_log:
        time.sleep(comm.rank*1)
        print(
            "setup_dist before init_process_group",
            "backend",backend,
            "cuda availability",th.cuda.is_available(),
            "MASTER_ADDR",os.environ["MASTER_ADDR"],
            "MASTER_PORT",os.environ["MASTER_PORT"],
            "WORLD_SIZE",os.environ["WORLD_SIZE"],
            "CUDA_VISIBLE_DEVICES",os.environ["CUDA_VISIBLE_DEVICES"],
            "torch.cuda.current_device()",th.cuda.current_device(),
            "torch.cuda.device_count()",th.cuda.device_count(),
            "GPUS_PER_NODE",GPUS_PER_NODE
        )
    dist.init_process_group(backend=backend, init_method="env://")
    if debug_log:
        print(
            "setup_dist after init_process_group",
            "backend",backend,
            "cuda availability",th.cuda.is_available(),
            "MASTER_ADDR",os.environ["MASTER_ADDR"],
            "MASTER_PORT",os.environ["MASTER_PORT"],
            "WORLD_SIZE",os.environ["WORLD_SIZE"],
            "CUDA_VISIBLE_DEVICES",os.environ["CUDA_VISIBLE_DEVICES"],
            "torch.cuda.current_device()",th.cuda.current_device(),
            "torch.cuda.device_count()",th.cuda.device_count(),
            "GPUS_PER_NODE",GPUS_PER_NODE
        )

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, debug_log = False, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            if debug_log:
                print("bcast from rank 0",i,i+chunk_size)
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
