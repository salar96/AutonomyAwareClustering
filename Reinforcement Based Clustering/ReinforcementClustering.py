import numpy as np
from scipy.spatial.distance import cdist


def epsilon_schedule(t=0):
    return 0.1  # return 1 / (1 + t)


def alpha_schedule(t=0):
    return 0.001


def rho_schedule(i=0):
    return 1.0


def d_t(x, y):
    return 0.5 * (np.linalg.norm(x - y)) ** 2


def prob_p_kji(N, M):
    p_kji = np.full((M, M, N), 0.0 / (M - 1))  # Default: uniform for k ≠ j

    for i in range(N):
        for j in range(M):
            p_kji[j, j, i] = 1.0  # Set p(k = j | j, i)

    return p_kji


def reinforcement_clustering(
    beta_min, beta_max, tau, M, X, T_p, episodes=100, GD_iter=1000, tol=1e-4
):

    N, d = X.shape
    beta = beta_min
    pi = np.full((N, M), 1 / M)  # policy
    centroid = np.mean(X, axis=0)
    Y = np.tile(centroid, (M, 1))  # Duplicate the centroid M times
    Y_s = [Y]  # List to keep track of centroids
    assignment_list = [np.zeros(N)]  # List to keep track of assignments
    # prob = prob_p_kji(N, M)
    prob = T_p
    t = 0  # time step (used for schedules)

    buffer = np.zeros((N, M, M))  # keep memory of interactions

    while beta <= beta_max:
        print(f"Beta: {beta:.3f}")
        d_bar = cdist(X, Y, metric="sqeuclidean") / 2  # shape (N, M)
        for _ in range(episodes):  # Outer convergence loop
            for i in range(N):
                j = np.random.choice(M, p=pi[i])  # this should be greedy
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

        beta *= tau  # annealing
        Y_s.append(Y)
        assignment_list.append(np.argmax(pi, axis=1))

    return assignment_list, Y_s
