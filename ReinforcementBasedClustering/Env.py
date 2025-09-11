# Creating environments for stochastic clustering
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist


class ClusteringEnvNumpy:
    def __init__(self, n_data, n_clusters, n_features, parametrized, T_p=None, seed=0):
        np.random.seed(seed)
        self.n_data = n_data
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.T_p = np.zeros(
            (n_clusters, n_clusters, n_data)
        )  # Transition probabilities p(k = j | j, i)
        self.parametrized = parametrized
        self.T_p = T_p
        self.prob = None
        _ = self.return_probabilities(None, None)
    def return_probabilities(self, X, Y, gamma=1.0):
        # for the case we want to have direct access to probabilities
        if self.parametrized:
            prob = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean") ** 2)  # (M, M)
            prob = prob / prob.sum(axis=0, keepdims=True)
            # repeat prob on axis = 2 N times
            prob = np.repeat(prob[:, :, np.newaxis], self.n_data, axis=2)  # (M, M, N)
        else:
            if self.T_p is not None:
                prob = self.T_p
            else:
                prob = np.full(
                    (self.n_clusters, self.n_clusters, self.n_data),
                    0.0 / (self.n_clusters - 1),
                )  # Default: uniform for k ≠ j

                for i in range(self.n_data):
                    for j in range(self.n_clusters):
                        prob[j, j, i] = 1.0  # Set p(k = j | j, i)
        self.prob = prob
        return prob

    def step(self, i, j, X=None, Y=None):
        if self.parametrized:
            self.return_probabilities(X, Y)
        k = np.random.choice(self.n_clusters, p=self.prob[:, j, i])
        return k
