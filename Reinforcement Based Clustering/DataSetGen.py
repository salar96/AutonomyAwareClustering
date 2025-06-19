import numpy as np


def generate_cluster_data(N, K, P, random_seed=None):
    """
    Generate N 2D points clustered into K groups with variance P.

    Parameters:
    - N: int, total number of points
    - K: int, number of clusters
    - P: float, variance within each cluster
    - random_seed: int or None, for reproducibility

    Returns:
    - data: np.ndarray of shape (N, 2), the generated 2D points
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    points_per_cluster = [N // K] * K
    for i in range(N % K):
        points_per_cluster[i] += 1

    # Randomly place cluster centers in a bounding box
    cluster_centers = np.random.uniform(-10, 10, size=(K, 2))

    data = []
    for i in range(K):
        # Generate points around each center
        cluster_points = (
            np.random.randn(points_per_cluster[i], 2) * np.sqrt(P) + cluster_centers[i]
        )
        data.append(cluster_points)

    return np.vstack(data)
