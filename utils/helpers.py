"""
Helper utility functions
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def seed_worker(worker_id):
    """Seed worker for DataLoader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Total number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def create_directories(*dirs):
    """Create multiple directories if they don't exist"""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def crop_img(image, base=16):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]