import torch

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