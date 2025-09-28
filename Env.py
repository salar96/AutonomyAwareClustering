# Creating environments for stochastic clustering
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist


class ClusteringEnvNumpy:
    """
    A NumPy implementation of a clustering environment for reinforcement learning.
    This environment models a Markov Decision Process where:
    - States are data points (i) that need to be assigned to clusters
    - Actions are cluster assignments (j)
    - Transitions move to new clusters (k) based on transition probabilities
    The transition probabilities p(k|j,i) can be either:
    1. Parametrized based on distances between data points and cluster centers
    2. Fixed to predefined values
    Parameters
    ----------
    n_data : int
        Number of data points to cluster
    n_clusters : int
        Number of clusters
    n_features : int
        Dimensionality of the data points
    parametrized : bool
        If True, transition probabilities are computed from data and cluster distances
        If False, fixed transition probabilities are used
    T_p : ndarray, optional, shape (n_clusters, n_clusters, n_data)
        Fixed transition probabilities when parametrized=False
    kappa : float, default=0.3
        Exploration probability (kappa) - probability of transitioning to a different cluster
    gamma : float, default=0.0
        Weight for data-cluster distances d(i,k) in utility function
    zeta : float, default=1.0
        Weight for cluster-cluster distances d(j,k) in utility function
    T : float, default=1.0
        Softmax temperature parameter - controls randomness in transitions
    seed : int, default=0
        Random seed for reproducibility
    Methods
    -------
    return_probabilities(X, Y)
        Computes transition probabilities p(k|j,i) based on data points X and cluster centers Y
    step(i, j, X=None, Y=None)
        Performs one transition step from data point i and cluster j, returning new cluster k
    """

    def __init__(
        self,
        n_data,
        n_clusters,
        n_features,
        parametrized,
        T_p=None,
        kappa=0.3,
        gamma=0.0,
        zeta=1.0,
        T=1.0,
        seed=0,
    ):
        np.random.seed(seed)
        self.n_data = n_data
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.parametrized = parametrized
        self.kappa = kappa  # exploration probability (kappa)
        self.gamma = gamma  # weight for d(i,k)
        self.zeta = zeta  # weight for d(j,k)
        self.T = T  # softmax temperature
        self.T_p = T_p  # optional fixed transition probabilities
        self.prob = None
        _ = self.return_probabilities(None, None)

    def return_probabilities(self, X, Y):
        """
        Compute p(k|j,i).
        X : (n_data, n_features)   data points
        Y : (n_clusters, n_features) cluster centers
        """
        if self.parametrized and X is not None and Y is not None:
            # Pairwise squared distances
            d_ik = np.sum(
                (X[:, None, :] - Y[None, :, :]) ** 2, axis=2
            )  # (n_data, n_clusters)
            d_jk = np.sum(
                (Y[:, None, :] - Y[None, :, :]) ** 2, axis=2
            )  # (n_clusters, n_clusters)

            # utilities u_k(j,i) = zeta*d_jk + gamma*d_ik
            # Result shape: (k, j, i)
            u = (
                self.zeta * d_jk.T[:, :, None]  # (k,j,1)
                + self.gamma * d_ik.T[:, None, :]  # (k,1,i)
            )

            # mask True where k != j
            mask = ~np.eye(self.n_clusters, dtype=bool)[:, :, None]  # (k,j,1)

            # exponentiate only off-diagonal terms
            u_masked = np.where(mask, u, np.inf)  # (k,j,i) with inf where k == j
            u_masked_mins = np.min(u_masked, axis=0, keepdims=True)  # (1,j,i)
            # subtract mins for numerical stability
            exp_u = np.exp(-(u_masked - u_masked_mins) / self.T)

            # softmax over k dimension, but only for k ≠ j
            denom = exp_u.sum(axis=0, keepdims=True)  # (1, j, i)

            # kappa * normalized exp(-u/T) for k ≠ j
            prob = np.where(mask, self.kappa * exp_u / denom, 0.0)

            # diagonal entries k == j get 1 - kappa
            diag = np.arange(self.n_clusters)
            prob[diag, diag, :] = 1.0 - self.kappa

        else:
            if self.T_p is not None:
                prob = self.T_p
            else:
                prob = np.full(
                    (self.n_clusters, self.n_clusters, self.n_data),
                    0.0 / (self.n_clusters - 1),
                )
                for i in range(self.n_data):
                    for j in range(self.n_clusters):
                        prob[j, j, i] = 1.0
        self.prob = prob
        return prob

    def step(self, i, j, X=None, Y=None):
        if self.parametrized:
            self.return_probabilities(X, Y)
        k = np.random.choice(self.n_clusters, p=self.prob[:, j, i])
        return k


class ClusteringEnvTorch:
    """
    PyTorch implementation of the clustering environment.

    Parameters
    ----------
    n_data : int
    n_clusters : int
    n_features : int
    parametrized : bool
        If True, transition probabilities are computed from data and cluster distances.
        If False, fixed transition probabilities are used.
    T_p : torch.Tensor, optional, shape (n_clusters, n_clusters, n_data)
        Fixed transition probabilities when parametrized=False.
    kappa : float, default=0.3
        Exploration probability.
    gamma : float, default=0.0
        Weight for data-cluster distances d(i,k).
    zeta : float, default=1.0
        Weight for cluster-cluster distances d(j,k).
    T : float, default=1.0
        Softmax temperature.
    seed : int, default=0
        Random seed.
    device : torch.device or str, default="cpu"
    """

    def __init__(
        self,
        n_data,
        n_clusters,
        n_features,
        parametrized,
        T_p=None,
        kappa=0.3,
        gamma=0.0,
        zeta=1.0,
        T=1.0,
        seed=0,
        device="cpu",
    ):
        torch.manual_seed(seed)
        self.n_data = n_data
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.parametrized = parametrized
        self.kappa = kappa
        self.gamma = gamma
        self.zeta = zeta
        self.T = T
        self.device = torch.device(device)
        self.T_p = T_p.to(self.device) if T_p is not None else None
        self.prob = None
        _ = self.return_probabilities(None, None)  # initialize prob

    @torch.no_grad()  # no need to track gradients here
    def return_probabilities(self, X, Y):
        """
        Compute p(k|j,i).
        X : (n_data, n_features)
        Y : (n_clusters, n_features)
        """
        if self.parametrized and X is not None and Y is not None:
            X = X.to(self.device)
            Y = Y.to(self.device)

            # pairwise squared distances
            d_ik = torch.sum(
                (X[:, None, :] - Y[None, :, :]) ** 2, dim=2
            )  # (n_data, n_clusters)
            d_jk = torch.sum(
                (Y[:, None, :] - Y[None, :, :]) ** 2, dim=2
            )  # (n_clusters, n_clusters)

            # utilities u_k(j,i) = zeta*d_jk + gamma*d_ik  -> (k, j, i)
            u = self.zeta * d_jk.T[:, :, None] + self.gamma * d_ik.T[:, None, :]

            mask = ~torch.eye(self.n_clusters, dtype=torch.bool, device=self.device)[
                :, :, None
            ]
            u_masked = torch.where(
                mask, u, torch.tensor(float("inf"), device=self.device)
            )
            u_masked_mins = torch.min(u_masked, dim=0, keepdim=True).values
            exp_u = torch.exp(-(u_masked - u_masked_mins) / self.T)
            denom = exp_u.sum(dim=0, keepdim=True)  # (1, j, i)

            prob = torch.where(mask, self.kappa * exp_u / denom, torch.zeros_like(u))
            diag = torch.arange(self.n_clusters, device=self.device)
            prob[diag, diag, :] = 1.0 - self.kappa

        else:
            if self.T_p is not None:
                prob = self.T_p
            else:
                prob = torch.full(
                    (self.n_clusters, self.n_clusters, self.n_data),
                    0.1 / (self.n_clusters - 1),
                    device=self.device,
                )
                for i in range(self.n_data):
                    for j in range(self.n_clusters):
                        prob[j, j, i] = 0.9

        self.prob = prob
        return prob

    @torch.no_grad()  # no need to track gradients here
    def step(self, batch_indices_all, idx, B, S, mc_samples, X=None, Y=None):
        """
        Batched Monte-Carlo sampling of next clusters.

        Parameters
        ----------
        batch_indices_all : (B, S) LongTensor
            Indices of data points i.
        idx : (B, S) LongTensor
            Current cluster assignments j.
        B : int
            Batch size.
        S : int
            Number of samples in each batch.
        mc_samples : int
            Number of Monte-Carlo samples per (B,S).

        Returns
        -------
        realized_clusters : (B, S, mc_samples) LongTensor
        """

        M = self.n_clusters
        if self.parametrized:
            self.return_probabilities(X, Y)  # update self.prob if needed
        probs = self.prob.to(self.device)

        m_idx = torch.arange(M, device=self.device).view(1, 1, M).expand(B, S, M)
        i_idx = batch_indices_all.unsqueeze(-1).expand(B, S, M)
        j_idx = idx.unsqueeze(-1).expand(B, S, M)

        prob_matrix = probs[m_idx, j_idx, i_idx]  # (B, S, M)
        flat_probs = prob_matrix.reshape(-1, M)  # (B*S, M)

        assert torch.all(torch.isfinite(flat_probs)), "NaN/Inf in probs"
        assert torch.all(flat_probs >= 0), "Negative probs"
        row_sums = flat_probs.sum(dim=1)
        assert torch.all(row_sums > 0), "Row with zero total probability"
        
        realized_clusters = torch.multinomial(flat_probs, mc_samples, replacement=True)
        realized_clusters = realized_clusters.view(B, S, mc_samples)
        return realized_clusters


if __name__ == "__main__":
    # test the torch environment
    N = 100  # number of data points
    M = 5  # number of clusters
    d = 2  # number of features
    env = ClusteringEnvTorch(
        n_data=N, n_clusters=M, n_features=d, parametrized=True, device="cuda"
    )
    X = torch.randn(N, d)
    Y = torch.randn(M, d)
    prob = env.return_probabilities(X, Y)
    print("Transition probabilities shape:", prob.shape)  # should be (M, M, N)
    bathch_indices_all = torch.randint(0, N, (10, 4)).long().to("cuda")
    idx = torch.randint(0, M, (10, 4)).long().to("cuda")
    next_clusters = env.step(bathch_indices_all, idx, 10, 4, 3)
    print("\033[93mNext clusters shape:", next_clusters.shape, "\033[0m")  # should be (10, 4, 3)
