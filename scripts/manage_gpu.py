# manage_gpu.py
import torch

def free_gpu_memory():
    torch.cuda.empty_cache()