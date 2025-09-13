# Creating environments for stochastic clustering
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist


class ClusteringEnvNumpy:
    def __init__(self, n_data, n_clusters, n_features,
                 parametrized, T_p=None, seed=0):
        np.random.seed(seed)
        self.n_data = n_data
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.parametrized = parametrized
        self.T_p = T_p  # optional fixed transition probabilities
        self.prob = None
        _ = self.return_probabilities(None, None)

    def return_probabilities(self, X, Y, gamma=1.0,
                             eps=0.1, zeta=1.0, T=1.0):
        """
        Compute p(k|j,i).
        X : (n_data, n_features)   data points
        Y : (n_clusters, n_features) cluster centers
        gamma : weight for d(i,k)
        zeta  : weight for d(j,k)
        eps   : exploration probability (epsilon)
        T     : softmax temperature
        """
        if self.parametrized and X is not None and Y is not None:
            # Pairwise squared distances
            d_ik = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2)      # (n_data, n_clusters)
            d_jk = np.sum((Y[:, None, :] - Y[None, :, :])**2, axis=2)      # (n_clusters, n_clusters)

            # utilities u_k(j,i) = zeta*d_jk + gamma*d_ik
            # Result shape: (k, j, i)
            u = (
                zeta  * d_jk.T[:, :, None] +          # (k,j,1)
                gamma * d_ik.T[:, None, :]            # (k,1,i)
            )

            # mask True where k != j
            mask = ~np.eye(self.n_clusters, dtype=bool)[:, :, None]  # (k,j,1)

            # exponentiate only off-diagonal terms
            u_masked = np.where(mask, u, np.inf) # (k,j,i) with inf where k == j
            
            exp_u    = np.exp(-u_masked / T)

            # softmax over k dimension, but only for k ≠ j
            denom = exp_u.sum(axis=0, keepdims=True)  # (1, j, i)

            # epsilon * normalized exp(-u/T) for k ≠ j
            prob = np.where(mask, eps * exp_u / denom, 0.0)

            # diagonal entries k == j get 1 - eps
            diag = np.arange(self.n_clusters)
            prob[diag, diag, :] = 1.0 - eps

        else:
            if self.T_p is not None:
                prob = self.T_p
            else:
                prob = np.full(
                    (self.n_clusters, self.n_clusters, self.n_data),
                    0.1 / (self.n_clusters - 1),
                )
                for i in range(self.n_data):
                    for j in range(self.n_clusters):
                        prob[j, j, i] = 0.9
        self.prob = prob
        return prob

    def step(self, i, j, X=None, Y=None):
        if self.parametrized:
            self.return_probabilities(X, Y)
        k = np.random.choice(self.n_clusters, p=self.prob[:, j, i])
        return k

if __name__ == "__main__":
    # Simple test
    env = ClusteringEnvNumpy(n_data=5, n_clusters=3, n_features=2,
                             parametrized=True)
    X = np.random.randn(5, 2)
    Y = np.random.randn(3, 2)
    print("Data points:\n", X)
    print("Cluster centers:\n", Y)
    print("Transition probabilities p(k|j,i):\n", env.return_probabilities(X, Y))
    for i in range(5):
        for j in range(3):
            k = env.step(i, j, X, Y)
            print(f"From data point {i} and cluster {j}, moved to cluster {k}")
