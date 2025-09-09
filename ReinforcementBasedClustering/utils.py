import torch
import random
import numpy as np


def d_t(x, y):
    """
    Squared Euclidean distance (scaled by 0.5) between batched x and y.

    Args:
        x: Tensor (..., input_dim)
        y: Tensor (..., input_dim)
    Returns:
        Tensor (...)  # one distance per pair
    """
    return 0.5 * torch.sum((x - y) ** 2, dim=-1)


def set_seed(seed: int = 42):
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed fixed to {seed}]")
