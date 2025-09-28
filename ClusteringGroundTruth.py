import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def compute_cluster_centers(X, rho, pi, p_l_given_ji, eps=1e-12):
    """
    Compute Y (M x d) where
      y_l = sum_{i,j} rho[i] * p_l_given_ji[l,j,i] * pi[i,j] * x_i
            ------------------------------------------------------
            sum_{i,j} rho[i] * p_l_given_ji[l,j,i] * pi[i,j]

    Parameters
    ----------
    X : (N, d) array
    rho : (N,) array
    pi : (N, M) array          # pi[i,j] = π(j|i)
    p_l_given_ji : (M, M, N)   # p_l_given_ji[l,j,i] = p(l|j,i)
    eps : small float to avoid division by zero (default 1e-12)

    Returns
    -------
    Y : (M, d) array of cluster centers
    """
    N, d = X.shape
    M = pi.shape[1]
    assert rho.shape == (N,)
    assert pi.shape == (N, M)
    assert p_l_given_ji.shape == (M, M, N)

    # Precompute rho * pi (N,M)
    R = rho[:, None] * pi

    # Numerator: ∑_{l,j,i} R[i,j] * p[l,j,i] * X[i]
    numerator = np.einsum("ij,lji,id->ld", R, p_l_given_ji, X)

    # Denominator: ∑_{l,j,i} R[i,j] * p[l,j,i]
    denom = np.einsum("ij,lji->l", R, p_l_given_ji)

    denom_safe = denom + (denom == 0) * eps
    Y = numerator / denom_safe[:, None]

    return Y

def free_energy(X, Y, rho, env, beta):
    M = Y.shape[0]
    N = X.shape[0]
    # update policy
    p_l_given_ji = env.return_probabilities(X, Y)  # (M,M,N)
    D = cdist(X, Y, "sqeuclidean", out=None)
    D_bar = np.einsum("il,lji->ij", D, p_l_given_ji) # (N,M)
    mins = np.min(D_bar, axis=1, keepdims=True) # (N,1)
    # calculating free energy (with numerical trick to avoid log(0) issues)
    F = -1/beta * np.log(np.sum(np.exp(-beta * (D_bar-mins)), axis=1, keepdims=True)) +  mins # (N,1)
    F = np.sum(rho[:, None] * F) # scalar
    return F

def distortion(X, Y, rho, pi, env):
    M = Y.shape[0]
    N = X.shape[0]
    # update policy
    p_l_given_ji = env.return_probabilities(X, Y)  # (M,M,N)
    D = cdist(X, Y, "sqeuclidean", out=None)
    D_bar = np.einsum("il,lji->ij", D, p_l_given_ji) # (N,M)
    D_avg = np.sum(rho[:, None] * np.sum(pi * D_bar, axis=1, keepdims=True)) # scalar
    return D_avg

def cluster_gt(X, Y, rho, env, beta_min=0.1, beta_max=10000.0, tau=1.5, verbose=False):
    M = Y.shape[0]
    N = X.shape[0]
    beta = beta_min
    Y_list = []
    pi_list = []
    Betas = []
    while beta <= beta_max:
        
        for _ in range(100):
            # update policy
            p_l_given_ji = env.return_probabilities(X, Y)  # (M,M,N)
            D = cdist(X, Y, "sqeuclidean", out=None)
            D_bar = np.einsum("il,lji->ij", D, p_l_given_ji)
            mins = np.min(D_bar, axis=1, keepdims=True)
            D_bar_normalized = D_bar - mins
            pi = np.exp(-beta * D_bar_normalized)
            pi /= np.sum(pi, axis=1, keepdims=True)
            # update cluster centers
            Y = compute_cluster_centers(X, rho, pi, p_l_given_ji, eps=1e-12)

            
        distortion_val = distortion(X, Y, rho, pi, env)
        if verbose:
            print(f"beta: {beta:.2f}, distortion: {distortion_val:.4f}")
        Y_list.append(Y)
        pi_list.append(pi)
        Betas.append(beta)
        beta *= tau
        Y += np.random.randn(M, 2) * 0.001  # Add small noise to avoid local minima
    return Y, pi, Y_list, pi_list, Betas

def cluster_gt_solver(X, Y, rho, env, beta_min=0.1, beta_max=10000.0, tau=1.5, verbose=False):
    M = Y.shape[0]
    N = X.shape[0]
    d = X.shape[1]
    beta = beta_min

    Y_list = []
    pi_list = []
    Betas = []
    while beta <= beta_max:
      
        # use scipy optimize to minimize free enrgy with respect to Y

        def objective(Y_flat):
            Y = Y_flat.reshape(M, d)
            return free_energy(X, Y, rho, env, beta)

        # Define bounds to keep values in the unit box [0,1]
        bounds = [(0, 1) for _ in range(M * d)]
        
        # Use L-BFGS-B method which supports bounds
        res = minimize(objective, Y.flatten(), method="powell", bounds=bounds)
        Y = res.x.reshape(M, d)
        distances = cdist(X, Y, "sqeuclidean")
        pi = np.exp(-beta * (distances - np.min(distances, axis=1, keepdims=True)))
        pi /= np.sum(pi, axis=1, keepdims=True)
        distortion_val = distortion(X, Y, rho, pi, env)
        if verbose:
            print(f"beta: {beta:.2f}, distortion: {distortion_val:.4f}")
        Y_list.append(Y)
        pi_list.append(pi)
        Betas.append(beta)
        beta *= tau
        Y += np.random.randn(M, 2) * 0.001  # Add small noise to avoid local minima
    return Y, pi, Y_list, pi_list, Betas