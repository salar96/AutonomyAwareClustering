import numpy as np
from scipy.spatial.distance import cdist
from Env import ClusteringEnvNumpy


def epsilon_schedule(beta):
    return 0.1  # return 1 / (1 + t)


def alpha_schedule(t=0):
    return 0.0001


def rho_schedule(t=0):
    return 0.1


def d_t(x, y):
    return 0.5 * (np.linalg.norm(x - y)) ** 2


def derivative_vectorized(Y, X, pi, transition_prob):
    """
    Vectorized implementation.
    """
    M, d = Y.shape
    N = X.shape[0]
    prob = transition_prob.transpose(2, 1, 0)  # (M, M, N)
    # weight[l, j, i] = pi[i,j] * prob[l,j,i]
    weight = pi.T[None, :, :] * prob  # (M, M, N)

    # sum over j → (M, N)
    W = weight.sum(axis=1)

    # term1 = coeff * Y
    term1 = (W.sum(axis=1)[:, None]) * Y

    # term2 = - sum_{i} W[l,i] * X[i]
    term2 = -(W @ X)

    return term1 + term2


def derivative_sampled(Y, X, pi, prob):
    """
    Stochastic gradient version: sample one k ~ prob[:, j, i] for each (i,j).

    Y: (M,d)
    X: (N,d)
    pi: (N,M)
    prob: (M,M,N)   # prob[l,j,i]

    Returns:
    D_Y: (M,d)
    """
    M, d = Y.shape
    N = X.shape[0]

    # Flatten (i,j) pairs into a single dimension for vectorized sampling
    probs_flat = prob.reshape(M, -1).T  # shape (N*M, M)
    ks = np.array([np.random.choice(M, p=p) for p in probs_flat])
    ks = ks.reshape(N, M)  # back to shape (N, M)

    # Prepare accumulation
    D_Y = np.zeros((M, d))

    # For each cluster l, gather contributions where ks == l
    for l in range(M):
        mask = ks == l  # shape (N, M)
        weights = (pi * mask).sum(axis=1)  # sum over j → shape (N,)
        D_Y[l] = (weights[:, None] * (Y[l] - X)).sum(axis=0)

    return D_Y


def derivative_sampled_fast(Y, X, pi, prob):
    """
    Stochastic gradient version: sample one k ~ prob[:, j, i] for each (i,j).

    Y: (M,d)
    X: (N,d)
    pi: (N,M)
    prob: (M,M,N)   # prob[l,j,i]

    Returns:
    D_Y: (M,d)
    """
    M, d = Y.shape
    N = X.shape[0]

    # --- Step 1: sample all ks in one shot ---
    probs_flat = prob.reshape(M, -1).T  # (N*M, M)
    rand = np.random.rand(probs_flat.shape[0], 1)
    cdf = np.cumsum(probs_flat, axis=1)
    ks_flat = (rand < cdf).argmax(axis=1)  # indices sampled
    ks = ks_flat.reshape(N, M)  # (N,M)

    # --- Step 2: accumulate ---
    i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(M), indexing="ij")
    sampled_weights = np.zeros((N, M, M))
    sampled_weights[i_idx, j_idx, ks] = pi[i_idx, j_idx]

    W = sampled_weights.sum(axis=1)  # (N,M)

    term1 = (W.sum(axis=0)[:, None]) * Y
    term2 = -(W.T @ X)

    return term1 + term2


def reinforcement_clustering(
    beta_min,
    beta_max,
    tau,
    M,
    X,
    env: ClusteringEnvNumpy,
    episodes=100,
    GD_iter=1000,
    tol=1e-3,
    perturbation=0.01,
):

    N, d = X.shape
    beta = beta_min
    pi = np.full((N, M), 1 / M)  # policy
    centroid = np.mean(X, axis=0)
    Y = (
        np.tile(centroid, (M, 1)) + np.random.randn(M, d) * perturbation
    )  # Initialize centroids
    Y_s = [Y]  # List to keep track of centroids
    assignment_list = [np.zeros(N)]  # List to keep track of assignments
    

    buffer = np.zeros((N, M, M))  # keep memory of interactions

    while beta <= beta_max:
        print(f"Beta: {beta:.3e}")
        d_bar = cdist(X, Y, metric="sqeuclidean") / 2  # shape (N, M)
        t = 0
        buffer.fill(0)  # reset buffer
        for _ in range(episodes):  # Outer convergence loop
            for i in range(N):
                epsilon = epsilon_schedule(beta)
                if np.random.random() < epsilon:
                    # Explore: select a random action
                    j = np.random.randint(M)
                else:
                    # Exploit: select the action with highest probability
                    j = np.argmax(pi[i])
                k = env.step(i, j, X, Y)  # sample according to env
                buffer[i, j, k] += 1
                rho = rho_schedule(t)
                d_bar[i, j] = rho * d_bar[i, j] + (1 - rho) * d_t(X[i], Y[k])

        d_mins = np.min(d_bar, axis=1, keepdims=True)
        pi = np.exp(-beta * (d_bar - d_mins))
        pi /= pi.sum(axis=1, keepdims=True)  # shape (N, M)
        transition_prob = buffer / (
            np.sum(buffer, axis=2, keepdims=True) + 1e-8
        )  # shape (N, M, M)
        
        derivs = np.zeros_like(Y)  # shape (M , 2)
        pi_p_all = np.sum(transition_prob * pi[:, :, None], axis=1)
        # for _ in range(GD_iter):  # Inner convergence loop

        #     diff = Y[:, None, :] - X[None, :, :]
        #     derivs = np.sum(diff * pi_p_all.T[:, :, None], axis=1)
        #     Y = Y - alpha_schedule(t) * derivs
        #     if np.linalg.norm(derivs) < tol:
        #         break
        #     t += 1  # increment time step
        T_P = env.return_probabilities(X, Y)
        for _ in range(GD_iter):
            derivs = derivative_sampled_fast(Y, X, pi, T_P)
            Y = Y - alpha_schedule(t) * derivs
            if np.linalg.norm(derivs) < tol:
                break
            t += 1

        beta *= tau  # annealing
        Y_s.append(Y)
        Y += np.random.randn(M, d) * perturbation  # add perturbation
        assignment_list.append(pi)

    return assignment_list, Y_s
