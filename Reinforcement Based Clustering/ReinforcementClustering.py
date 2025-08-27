import numpy as np
from scipy.spatial.distance import cdist


def epsilon_schedule(t=0):
    return 0.1 # return 1 / (1 + t)


def alpha_schedule(t=0):
    return 0.001


def rho_schedule(i=0):
    return 1.0


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
    probs_flat = prob.reshape(M, -1).T   # shape (N*M, M)
    ks = np.array([np.random.choice(M, p=p) for p in probs_flat])
    ks = ks.reshape(N, M)  # back to shape (N, M)

    # Prepare accumulation
    D_Y = np.zeros((M, d))

    # For each cluster l, gather contributions where ks == l
    for l in range(M):
        mask = (ks == l)                       # shape (N, M)
        weights = (pi * mask).sum(axis=1)      # sum over j → shape (N,)
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
    probs_flat = prob.reshape(M, -1).T   # (N*M, M)
    rand = np.random.rand(probs_flat.shape[0], 1)
    cdf = np.cumsum(probs_flat, axis=1)
    ks_flat = (rand < cdf).argmax(axis=1)   # indices sampled
    ks = ks_flat.reshape(N, M)              # (N,M)

    # --- Step 2: accumulate ---
    i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(M), indexing="ij")
    sampled_weights = np.zeros((N, M, M))
    sampled_weights[i_idx, j_idx, ks] = pi[i_idx, j_idx]

    W = sampled_weights.sum(axis=1)   # (N,M)

    term1 = (W.sum(axis=0)[:, None]) * Y
    term2 = -(W.T @ X)

    return term1 + term2

def prob_p_kji(N, M):
    p_kji = np.full((M, M, N), 0.0 / (M - 1))  # Default: uniform for k ≠ j

    for i in range(N):
        for j in range(M):
            p_kji[j, j, i] = 1.0  # Set p(k = j | j, i)

    return p_kji


def reinforcement_clustering(
    beta_min,
    beta_max,
    tau,
    M,
    X,
    T_p,
    episodes=100,
    GD_iter=10000,
    tol=1e-3,
    perturbation=0.1,
    parametrized=False,
):

    N, d = X.shape
    beta = beta_min
    pi = np.full((N, M), 1 / M)  # policy
    centroid = np.mean(X, axis=0)
    Y = np.tile(centroid, (M, 1))  + np.random.randn(M, d) * perturbation  # Initialize centroids
    Y_s = [Y]  # List to keep track of centroids
    assignment_list = [np.zeros(N)]  # List to keep track of assignments
    # prob = prob_p_kji(N, M)
    if not parametrized:
        prob = T_p
    t = 0  # time step (used for schedules)

    buffer = np.zeros((N, M, M))  # keep memory of interactions
    gamma = 10.0
    while beta <= beta_max:
        print(f"Beta: {beta:.3f}")
        d_bar = cdist(X, Y, metric="sqeuclidean") / 2  # shape (N, M)
        if parametrized:
            prob = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean") ** 2)  # (M, M)
            prob = prob / prob.sum(axis=0, keepdims=True)
            # repeat prob on axis = 2 N times
            prob = np.repeat(prob[:, :, np.newaxis], N, axis=2)  # (M, M, N)
        buffer.fill(0)  # reset buffer
        for _ in range(episodes):  # Outer convergence loop
            for i in range(N):
                j = np.argmax(pi[i])  # greedy selection (highest probability)
                k = np.random.choice(M, p=prob[:, j, i])
                buffer[i, j, k] += 1
                eps = epsilon_schedule(t)
                d_bar[i, j] = eps * d_bar[i, j] + (1 - eps) * d_t(X[i], Y[k])

        d_mins = np.min(d_bar, axis=1, keepdims=True)
        pi = np.exp(-beta * (d_bar - d_mins))
        pi /= pi.sum(axis=1, keepdims=True)  # shape (N, M)
        transition_prob = buffer / (
            np.sum(buffer, axis=2, keepdims=True) + 1e-8
        )  # shape (N, M, M)

        derivs = np.zeros_like(Y)  # shape (M , 2)
        pi_p_all = np.sum(transition_prob * pi[:, :, None], axis=1)
        for _ in range(GD_iter):  # Inner convergence loop

            diff = Y[:, None, :] - X[None, :, :]
            derivs = np.sum(diff * pi_p_all.T[:, :, None], axis=1)
            Y = Y - alpha_schedule(t) * derivs
            if np.linalg.norm(derivs) < tol:
                break
            t += 1  # increment time step
        # for _ in range(GD_iter):
        #     derivs = derivative_vectorized(Y, X, pi, transition_prob)
        #     Y = Y - alpha_schedule(t) * derivs
        #     if np.linalg.norm(derivs) < tol:
        #         break
        #     t += 1

        beta *= tau  # annealing
        Y_s.append(Y)
        Y += np.random.randn(M, d) * perturbation  # add perturbation
        assignment_list.append(np.argmax(pi, axis=1))

    return assignment_list, Y_s
