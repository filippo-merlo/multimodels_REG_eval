# manage_gpu.py
import torch
import gc

def free_gpu_memory():
    # Run garbage collector to clear unused variables
    gc.collect()
    torch.cuda.empty_cache()