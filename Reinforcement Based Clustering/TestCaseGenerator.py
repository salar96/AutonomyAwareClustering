import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_RLClustering(idx):
    if idx == 1:
        data = pd.read_csv("streaming_users_dataset.csv")
        X = data[
            [
                "AvgWatchTime",
                "GenreDiversity",
                "PreferredGenreAction",
                "PreferredGenreDrama",
                "PreferredGenreComedy",
                "BingeFrequency",
                "SubscriptionTier",
            ]
        ].values

        K = 3
        N, d = X.shape
        T_P = np.zeros((K, K, N))

        for j in range(K):
            for i in range(N):
                probs = np.zeros(K)
                for l in range(K):
                    if l == j:
                        probs[l] = np.random.rand() * 0.3 + 0.7
                    else:
                        probs[l] = np.random.rand() * 0.1
                probs /= probs.sum()
                T_P[:, j, i] = probs

    elif idx == 2:
        C = np.array(
            [
                [0, 0],
                [1, 0],
                [0.5, 0.9],
                [5, 0],
                [6, 0],
                [5.5, 0.9],
                [2.5, 4.2],
                [3.5, 4.2],
                [3, 5],
            ]
        )
        np.random.seed(0)
        Np = 100
        X = np.vstack([np.random.normal(c, 0.125, (Np, 2)) for c in C])
        K = 9
        N, d = X.shape
        T_P = np.zeros((K, K, N))
        for j in range(K):
            for k in range(K):
                if j != k:
                    T_P[k, j, :] = 1 / (K * (K - 1))
            T_P[j, j, :] = (K - 1) / K

    elif idx == 3:
        C = np.array(
            [
                [0, 0],
                [2, 0],
                [1, 2],
                [4, 0],
                [6, 0],
                [5, 2],
                [2, 3.8],
                [4, 3.8],
                [3, 5.5],
                [3, 2],
            ]
        )
        np.random.seed(0)
        Np = 200
        X = np.vstack([np.random.normal(c, 0.175, (Np, 2)) for c in C])
        K = 10
        N, d = X.shape
        T_P = np.zeros((K, K, N))
        for j in range(K):
            for k in range(K):
                if j != k:
                    T_P[k, j, :] = 1 / (K * (K - 1))
            T_P[j, j, :] = (K - 1) / K

    elif idx == 4:
        C = np.array([[-4, -3], [4, -3], [4, 3], [-4, 3]])
        np.random.seed(0)
        sizes = [400, 500, 300, 600]
        stds = [0.4, 0.7, 0.5, 0.85]
        X = []
        C_labels = []
        for center, n, s, lab in zip(C, sizes, stds, range(1, 5)):
            pts = np.random.normal(center, s, (n, 2))
            X.append(pts)
            C_labels.extend([lab] * n)
        X = np.vstack(X)
        K = 4
        N, d = X.shape
        ############################################## CASE 1
        T_P = np.zeros((K, K, N))
        for j in range(K):
            for k in range(K):
                if j != k:
                    T_P[k, j, :] = 1 / (K * (K - 1))
            T_P[j, j, :] = (K - 1) / K
        ############################################## CASE 2
        # T_P = np.full((K, K, N), 0.0 / (K - 1))  # Default: uniform for k ≠ j

        # for i in range(N):
        #     for j in range(K):
        #         T_P[j, j, i] = 1.0  # Set p(k = j | j, i)
        ############################################## CASE 3

        plt.scatter(
            X[:, 0],
            X[:, 1],
            s=90,
            c=[0.7] * N,
            edgecolors=[0, 0.5, 0.5],
            linewidths=1.5,
        )
        plt.xlim([-7, 7])
        plt.ylim([-6, 6])
        plt.gca().set_aspect("equal", "box")
        plt.gca().tick_params(labelsize=25, width=1.0)
        plt.xticks([-5, 0, 5])
        plt.yticks([-5, 0, 5])
        # plt.savefig("Setup_2D.png", dpi=600)

    elif idx == 5:
        C11 = np.array([-8, -4])
        C21 = np.array([4, -4])
        C31 = np.array([4, 4])
        C41 = np.array([-8, 4])
        shifts = np.array([[0, 0], [3.5, 0], [0, 3.5], [3.5, 3.5]])
        C = np.vstack([c + shift for c in [C11, C21, C31, C41] for shift in shifts])
        np.random.seed(0)
        Np = 200
        X = np.vstack([np.random.normal(c, 0.25, (Np, 2)) for c in C])
        K = 9
        N, d = X.shape
        T_P = np.zeros((K, K, N))
        for j in range(K):
            for k in range(K):
                if j != k:
                    T_P[k, j, :] = 1 / (K * (K - 1))
            T_P[j, j, :] = (K - 1) / K

    elif idx == 6:
        C = np.array([[2, 4], [4, 7], [5, 5], [5, 3], [4, 1]])
        np.random.seed(0)
        Np = 200
        X = np.vstack([np.random.normal(c, 0.25, (Np, 2)) for c in C])
        K = 5
        N, d = X.shape
        T_P = np.zeros((K, K, N))
        for j in range(K):
            for k in range(K):
                if j != k:
                    T_P[k, j, :] = 1 / (K * (K - 1))
            T_P[j, j, :] = (K - 1) / K

    elif idx == 7:
        data = pd.read_csv("streaming_users_random_overrides.csv")
        X = data[
            [
                "AvgWatchTimeHrs",
                "GenreDiversity",
                "ActionPct",
                "DramaPct",
                "ComedyPct",
                "TrendingContentPct",
                "BingeFrequency",
                "LateNightViewingPct",
            ]
        ].values
        K = 3
        N, d = X.shape
        T_P = np.zeros((K, K, N))
        for j in range(K):
            for i in range(N):
                probs = np.zeros(K)
                for l in range(K):
                    if l == j:
                        probs[l] = np.random.rand() * 0.3 + 0.7
                    else:
                        probs[l] = np.random.rand() * 0.1
                probs /= probs.sum()
                T_P[:, j, i] = probs

    elif idx == 8:
        data = pd.read_csv("streaming_users_from_movielens.csv")
        X = data[
            [
                "WatchTime",
                "GenresWatched",
                "TrendingContentPct",
                "BingeFreq",
                "CompletionRate",
                "SubscriptionTier",
                "LateNightWatchPct",
            ]
        ].values
        K = 3
        N, d = X.shape
        T_P = np.zeros((K, K, N))
        for j in range(K):
            for i in range(N):
                probs = np.zeros(K)
                for l in range(K):
                    if l == j:
                        probs[l] = np.random.rand() * 0.3 + 0.7
                    else:
                        probs[l] = np.random.rand() * 0.1
                probs /= probs.sum()
                T_P[:, j, i] = probs

    elif idx == 9:
        data = pd.read_csv(
            "/home/amber-srivastava/OneDrive/Research/RL Clustering/Dataset/ml-20m/movielens20m_streaming_user_features.csv"
        )
        X = data[
            [
                "WatchTimeHours",
                "GenresWatched",
                "TrendingContentPct",
                "BingeFreq",
                "CompletionRate",
                "SubscriptionTier",
                "LateNightWatchPct",
            ]
        ].values
        K = 3
        N, d = X.shape
        T_P = np.zeros((K, K, N))
        for j in range(K):
            for i in range(N):
                probs = np.zeros(K)
                for l in range(K):
                    if l == j:
                        probs[l] = np.random.rand() * 0.3 + 0.7
                    else:
                        probs[l] = np.random.rand() * 0.1
                probs /= probs.sum()
                T_P[:, j, i] = probs

    return X, K, T_P, N, d

if __name__ == "__main__":
    idx = 4
    X, K, T_P, N, d = data_RLClustering(idx)
    print("Data shape:", X.shape)
    print("Number of clusters:", K)
    print("Transition probabilities shape:", T_P.shape)
    print("Number of samples:", N)
    print("Number of features:", d)
    print("shape of transition probabilities:", T_P.shape)
    print(T_P[:, :, 0])  # Print transition probabilities for the first sample