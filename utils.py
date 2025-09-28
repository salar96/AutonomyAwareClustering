import torch
import random
import numpy as np
from scipy.spatial.distance import cdist


def d_t(x, y):
    """
    Squared Euclidean distance between batched x and y.

    Args:
        x: Tensor (..., input_dim)
        y: Tensor (..., input_dim)
    Returns:
        Tensor (...)  # one distance per pair
    """
    return  torch.sum((x - y) ** 2, dim=-1)


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


def Chamfer_dist(Y_1 , Y_2):
    D = cdist(Y_1, Y_2, 'euclidean')
    chamfer = D.min(axis=1).mean() + D.min(axis=0).mean()
    all_pts = np.vstack([Y_1, Y_2])
    bbox_diag = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0) + 1e-12)
    return chamfer / bbox_diag * 100
def Hungarian_dist(Y_1 , Y_2):
    from scipy.optimize import linear_sum_assignment
    D = cdist(Y_1, Y_2, 'euclidean')
    row_ind, col_ind = linear_sum_assignment(D)
    hungarian = D[row_ind, col_ind].mean()
    all_pts = np.vstack([Y_1, Y_2])
    bbox_diag = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0) + 1e-12)
    return hungarian / bbox_diag * 100