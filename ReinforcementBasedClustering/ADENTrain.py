import torch
import utils


def TrainDbar(
    model,
    X,
    Y,
    device,
    epochs=1000,
    batch_size=32,
    num_samples_in_batch=128,
    lr=1e-4,
    weight_decay=1e-5,
    tol=1e-6,
    gamma=1000.0,
    probs=None,
    verbose=False,
):
    """
    Train ADEN model to learn expected distances.

    Args:
        model: PyTorch model (ADEN).
        X: Tensor of data points, shape (N, input_dim).
        Y: Tensor of cluster centroids, shape (M, input_dim).
        device: torch.device.
        epochs: Number of training epochs.
        batch_size: Number of batches.
        num_samples_in_batch: Samples per batch.
        lr: Learning rate.
        weight_decay: Optimizer weight decay.
        tol: Tolerance for early stopping.
        gamma: Transition probability scaling factor (used only if probs=None).
        probs: Optional tensor of shape (M, M, N), probabilities p(k | j, i).
    """

    N, input_dim = X.shape
    M = Y.shape[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # expand Y to batch dimension
    Y_batches = Y.unsqueeze(0).expand(batch_size, -1, -1).to(device).float()

    for param in model.parameters():
        param.requires_grad = True
    model.train()

    # default transition probs (M, M)
    if probs is None:
        transition_probs = torch.exp(-gamma * torch.cdist(Y, Y, p=2) ** 2)  # (M, M)
        transition_probs = transition_probs / transition_probs.sum(dim=-1, keepdim=True)
    else:
        # ensure correct dtype/device
        probs = probs.to(device).float()
    prev_mse_loss = float("inf")
    for epoch in range(epochs):
        # sample batches from X
        X_batches = torch.zeros(
            batch_size,
            num_samples_in_batch,
            input_dim,
            device=device,
            dtype=torch.float32,
        )
        batch_indices_all = []
        for i in range(batch_size):
            batch_indices = torch.randint(0, N, (num_samples_in_batch,), device=device)
            X_batches[i] = X[batch_indices]
            batch_indices_all.append(batch_indices)
        batch_indices_all = torch.stack(
            batch_indices_all, dim=0
        )  # (batch_size, num_samples_in_batch)

        # forward pass
        predicted_distances = model(
            X_batches, Y_batches
        )  # (batch_size, num_samples_in_batch, M)

        # closest cluster index for each point
        idx = torch.argmin(
            predicted_distances, dim=-1
        ).long()  # (batch_size, num_samples_in_batch)

        # mask only chosen cluster distances
        mask = torch.zeros_like(predicted_distances, dtype=torch.bool)
        mask.scatter_(2, idx.unsqueeze(2), 1)

        # --- Vectorized multinomial sampling ---
        if probs is None:
            # use default (M, M)
            prob_matrix = transition_probs[idx]  # (batch_size, num_samples_in_batch, M)
        else:
            B, S = batch_size, num_samples_in_batch

            # Expand batch and sample indices

            m_idx = torch.arange(M, device=device).view(1, 1, M).expand(B, S, M)

            # Gather i (data index) and j (chosen cluster)
            i_idx = batch_indices_all.unsqueeze(-1).expand(B, S, M)
            j_idx = idx.unsqueeze(-1).expand(B, S, M)

            # Advanced indexing into probs (M, M, N)
            prob_matrix = probs[m_idx, j_idx, i_idx]  # (B, S, M)

        # sample realized clusters
        realized_clusters = torch.multinomial(prob_matrix.view(-1, M), 1).view(
            batch_size, num_samples_in_batch
        )

        # gather the centroid coordinates of realized clusters
        realized_Y = Y_batches.gather(
            1, realized_clusters.unsqueeze(-1).expand(-1, -1, input_dim)
        )  # (batch_size, num_samples_in_batch, input_dim)

        # compute distances
        distances = utils.d_t(
            X_batches, realized_Y
        )  # (batch_size, num_samples_in_batch)

        # fill into D only at [batch, sample, idx]
        D = torch.zeros(batch_size, num_samples_in_batch, M, device=device)
        D.scatter_(2, idx.unsqueeze(-1), distances.unsqueeze(-1))

        # masked MSE loss
        mse_loss = torch.sum((D[mask] - predicted_distances[mask]) ** 2)

        if epoch % 1000 == 0 and verbose:
            print(f"[trainDbar] Epoch {epoch}, MSE Loss: {mse_loss.item():.3e}")

        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
        # stopping criterion based on change of loss
        if epoch > 0 and abs(mse_loss.item() - prev_mse_loss) / prev_mse_loss < tol:
            if verbose:
                print(f"Converged at epoch {epoch}, MSE Loss: {mse_loss.item():.3e}")
            break
        prev_mse_loss = mse_loss.item()


def TrainDbar_mc(
    model,
    X,
    Y,
    device,
    epochs=1000,
    batch_size=32,
    num_samples_in_batch=128,
    mc_samples=16,   # NEW: number of Monte Carlo samples per datapoint
    lr=1e-4,
    weight_decay=1e-5,
    tol=1e-6,
    gamma=1000.0,
    probs=None,
    verbose=False,
):
    """
    Train ADEN model to learn expected distances using Monte Carlo averaging.

    Args:
        mc_samples: Number of Monte Carlo samples per datapoint.
    """

    N, input_dim = X.shape
    M = Y.shape[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Y_batches = Y.unsqueeze(0).expand(batch_size, -1, -1).to(device).float()

    for param in model.parameters():
        param.requires_grad = True
    model.train()

    if probs is None:
        transition_probs = torch.exp(-gamma * torch.cdist(Y, Y, p=2) ** 2)  # (M, M)
        transition_probs = transition_probs / transition_probs.sum(dim=-1, keepdim=True)
    else:
        probs = probs.to(device).float()

    prev_mse_loss = float("inf")
    for epoch in range(epochs + 1):
        # sample batches from X
        X_batches = torch.zeros(
            batch_size,
            num_samples_in_batch,
            input_dim,
            device=device,
            dtype=torch.float32,
        )
        batch_indices_all = []
        for i in range(batch_size):
            batch_indices = torch.randint(0, N, (num_samples_in_batch,), device=device)
            X_batches[i] = X[batch_indices]
            batch_indices_all.append(batch_indices)
        batch_indices_all = torch.stack(batch_indices_all, dim=0)

        # forward pass
        predicted_distances = model(X_batches, Y_batches)  # (B, S, M)

        # closest cluster index for each point
        idx = torch.argmin(predicted_distances, dim=-1)  # (B, S)

        mask = torch.zeros_like(predicted_distances, dtype=torch.bool)
        mask.scatter_(2, idx.unsqueeze(2), 1)

        # --- Transition probabilities ---
        if probs is None:
            prob_matrix = transition_probs[idx]  # (B, S, M)
        else:
            B, S = batch_size, num_samples_in_batch
            m_idx = torch.arange(M, device=device).view(1, 1, M).expand(B, S, M)
            i_idx = batch_indices_all.unsqueeze(-1).expand(B, S, M)
            j_idx = idx.unsqueeze(-1).expand(B, S, M)
            prob_matrix = probs[m_idx, j_idx, i_idx]  # (B, S, M)

        # --- Monte Carlo averaging ---
        # (B*S, M)
        flat_probs = prob_matrix.view(-1, M)

        # draw mc_samples samples for each datapoint
        realized_clusters = torch.multinomial(flat_probs, mc_samples, replacement=True)
        realized_clusters = realized_clusters.view(batch_size, num_samples_in_batch, mc_samples)

        # gather centroids
        realized_Y = Y_batches.unsqueeze(2).expand(batch_size, M, mc_samples, input_dim)
        chosen_Y = realized_Y.gather(
            1,
            realized_clusters.unsqueeze(-1).expand(-1, -1, -1, input_dim)
        )  # (B, S, mc, input_dim)

        # compute distances and average
        distances = utils.d_t(
            X_batches.unsqueeze(2).expand(-1, -1, mc_samples, -1),  # (B, S, mc, dim)
            chosen_Y
        )  # (B, S, mc)
        mean_distances = distances.mean(dim=-1)  # (B, S)

        # fill into D only at [batch, sample, idx]
        D = torch.zeros(batch_size, num_samples_in_batch, M, device=device)
        D.scatter_(2, idx.unsqueeze(-1), mean_distances.unsqueeze(-1))

        # masked MSE loss
        mse_loss = torch.sum((D[mask] - predicted_distances[mask]) ** 2)

        if epoch % 1000 == 0 and verbose:
            print(f"[trainDbar] Epoch {epoch}, MSE Loss: {mse_loss.item():.3e}")

        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

        if epoch > 0 and abs(mse_loss.item() - prev_mse_loss) / prev_mse_loss < tol:
            if verbose:
                print(f"Converged at epoch {epoch}, MSE Loss: {mse_loss.item():.3e}")
            break
        prev_mse_loss = mse_loss.item()


def TrainDbar_RunningAvg(
    model,
    X,
    Y,
    device,
    epochs=1000,
    batch_size=32,
    num_samples_in_batch=128,
    lr=1e-4,
    weight_decay=1e-5,
    tol=1e-6,
    gamma=1000.0,
    alpha=0.1,   # smoothing factor for EMA
    probs=None,
    verbose=False,
):
    """
    Train ADEN model to learn expected distances via running average targets.
    """

    N, input_dim = X.shape
    M = Y.shape[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # expand Y to batch dimension
    Y_batches = Y.unsqueeze(0).expand(batch_size, -1, -1).to(device).float()

    for param in model.parameters():
        param.requires_grad = True
    model.train()

    # running average estimates of expected distances (N, M)
    running_D = torch.zeros(N, M, device=device)

    # default transition probs (M, M)
    if probs is None:
        transition_probs = torch.exp(-gamma * torch.cdist(Y, Y, p=2) ** 2)  # (M, M)
        transition_probs = transition_probs / transition_probs.sum(dim=-1, keepdim=True)
    else:
        probs = probs.to(device).float()

    prev_mse_loss = float("inf")
    for epoch in range(epochs):
        # sample batches from X
        X_batches = torch.zeros(
            batch_size, num_samples_in_batch, input_dim,
            device=device, dtype=torch.float32,
        )
        batch_indices_all = []
        for i in range(batch_size):
            batch_indices = torch.randint(0, N, (num_samples_in_batch,), device=device)
            X_batches[i] = X[batch_indices]
            batch_indices_all.append(batch_indices)
        batch_indices_all = torch.stack(batch_indices_all, dim=0)  # (B, S)

        # forward pass
        predicted_distances = model(X_batches, Y_batches)  # (B, S, M)

        # closest cluster index for each point
        idx = torch.argmin(predicted_distances, dim=-1).long()  # (B, S)

        # --- Vectorized multinomial sampling ---
        if probs is None:
            prob_matrix = transition_probs[idx]  # (B, S, M)
        else:
            B, S = batch_size, num_samples_in_batch
            m_idx = torch.arange(M, device=device).view(1, 1, M).expand(B, S, M)
            i_idx = batch_indices_all.unsqueeze(-1).expand(B, S, M)
            j_idx = idx.unsqueeze(-1).expand(B, S, M)
            prob_matrix = probs[m_idx, j_idx, i_idx]  # (B, S, M)

        # sample realized clusters
        realized_clusters = torch.multinomial(prob_matrix.view(-1, M), 1).view(
            batch_size, num_samples_in_batch
        )

        # gather the centroid coordinates of realized clusters
        realized_Y = Y_batches.gather(
            1, realized_clusters.unsqueeze(-1).expand(-1, -1, input_dim)
        )  # (B, S, d)

        # compute realized distances
        distances = utils.d_t(X_batches, realized_Y)  # (B, S)

        # --- Vectorized running average update ---
        flat_i = batch_indices_all.reshape(-1)   # (B*S,)
        flat_j = idx.reshape(-1)                 # (B*S,)
        flat_d = distances.reshape(-1)           # (B*S,)

        updates = torch.zeros_like(running_D)    # (N, M)
        counts  = torch.zeros_like(running_D)    # (N, M)

        updates.index_put_((flat_i, flat_j), flat_d, accumulate=True)
        counts.index_put_((flat_i, flat_j), torch.ones_like(flat_d), accumulate=True)

        avg_updates = updates / (counts + 1e-8)
        mask = counts > 0
        running_D[mask] = (1 - alpha) * running_D[mask] + alpha * avg_updates[mask]

        # --- Vectorized target construction ---
        # shape (B, S), gather running_D[i, j]
        targets = running_D[batch_indices_all, idx]  # (B, S)

        # fill into D at [b, s, j]
        D = torch.zeros(batch_size, num_samples_in_batch, M, device=device)
        D.scatter_(2, idx.unsqueeze(-1), targets.unsqueeze(-1))

        # masked MSE loss
        mask = torch.zeros_like(predicted_distances, dtype=torch.bool)
        mask.scatter_(2, idx.unsqueeze(2), 1)
        mse_loss = torch.sum((D[mask] - predicted_distances[mask]) ** 2)

        if epoch % 1000 == 0 and verbose:
            print(f"[TrainDbar_RunningAvg] Epoch {epoch}, MSE Loss: {mse_loss.item():.3e}")

        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

        # stopping criterion
        if epoch > 0 and abs(mse_loss.item() - prev_mse_loss) / prev_mse_loss < tol:
            if verbose:
                print(f"Converged at epoch {epoch}, MSE Loss: {mse_loss.item():.3e}")
            break
        prev_mse_loss = mse_loss.item()

def TrainDbar_Hybrid(
    model,
    X,
    Y,
    device,
    epochs=1000,
    batch_size=32,
    num_samples_in_batch=128,
    lr=1e-4,
    weight_decay=1e-5,
    tol=1e-6,
    gamma=1000.0,
    alpha=0.1,       # EMA smoothing factor
    L=8,             # number of env samples per datapoint (Monte Carlo averaging)
    probs=None,
    perturbation_std=0.01,  # small noise added to Y each iteration
    verbose=False,
):
    """
    Train ADEN model with hybrid strategy:
    - Multiple env samples per datapoint (Monte Carlo averaging)
    - Online running averages (EMA) for variance reduction
    """

    N, input_dim = X.shape
    M = Y.shape[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # expand Y to batch dimension
    Y_base = Y.unsqueeze(0).expand(batch_size, -1, -1).to(device).float()

    for param in model.parameters():
        param.requires_grad = True
    model.train()

    # running average estimates of expected distances (N, M)
    running_D = torch.zeros(N, M, device=device)

    # default transition probs (M, M)
    if probs is None:
        transition_probs = torch.exp(-gamma * torch.cdist(Y, Y, p=2) ** 2)  # (M, M)
        transition_probs = transition_probs / transition_probs.sum(dim=-1, keepdim=True)
    else:
        probs = probs.to(device).float()

    prev_mse_loss = float("inf")
    for epoch in range(epochs+1):
        # sample batches from X
        X_batches = torch.zeros(
            batch_size, num_samples_in_batch, input_dim,
            device=device, dtype=torch.float32,
        )
        Y_batches = Y_base + torch.randn_like(Y_base) * perturbation_std  # small perturbation
        batch_indices_all = []
        for i in range(batch_size):
            batch_indices = torch.randint(0, N, (num_samples_in_batch,), device=device)
            X_batches[i] = X[batch_indices]
            batch_indices_all.append(batch_indices)
        batch_indices_all = torch.stack(batch_indices_all, dim=0)  # (B, S)

        # forward pass
        predicted_distances = model(X_batches, Y_batches)  # (B, S, M)

        # closest cluster index for each point
        idx = torch.argmin(predicted_distances, dim=-1).long()  # (B, S)

        # --- Vectorized multinomial sampling ---
        if probs is None:
            prob_matrix = transition_probs[idx]  # (B, S, M)
        else:
            B, S = batch_size, num_samples_in_batch
            m_idx = torch.arange(M, device=device).view(1, 1, M).expand(B, S, M)
            i_idx = batch_indices_all.unsqueeze(-1).expand(B, S, M)
            j_idx = idx.unsqueeze(-1).expand(B, S, M)
            prob_matrix = probs[m_idx, j_idx, i_idx]  # (B, S, M)

        # --- Monte Carlo averaging ---
        realized_d_list = []
        for _ in range(L):
            # sample realized clusters
            realized_clusters = torch.multinomial(prob_matrix.view(-1, M), 1).view(
                batch_size, num_samples_in_batch
            )
            # gather centroid coordinates
            realized_Y = Y_batches.gather(
                1, realized_clusters.unsqueeze(-1).expand(-1, -1, input_dim)
            )
            # compute realized distances
            realized_d_list.append(utils.d_t(X_batches, realized_Y))  # (B, S)

        # average over L samples → Monte Carlo estimate
        distances = torch.stack(realized_d_list, dim=0).mean(0)  # (B, S)

        # --- Vectorized running average update ---
        flat_i = batch_indices_all.reshape(-1)   # (B*S,)
        flat_j = idx.reshape(-1)                 # (B*S,)
        flat_d = distances.reshape(-1)           # (B*S,)

        updates = torch.zeros_like(running_D)    # (N, M)
        counts  = torch.zeros_like(running_D)    # (N, M)

        updates.index_put_((flat_i, flat_j), flat_d, accumulate=True)
        counts.index_put_((flat_i, flat_j), torch.ones_like(flat_d), accumulate=True)

        avg_updates = updates / (counts + 1e-8)
        mask = counts > 0
        running_D[mask] = (1 - alpha) * running_D[mask] + alpha * avg_updates[mask]

        # --- Target construction from running averages ---
        targets = running_D[batch_indices_all, idx]  # (B, S)

        D = torch.zeros(batch_size, num_samples_in_batch, M, device=device)
        D.scatter_(2, idx.unsqueeze(-1), targets.unsqueeze(-1))

        # masked MSE loss
        mask = torch.zeros_like(predicted_distances, dtype=torch.bool)
        mask.scatter_(2, idx.unsqueeze(2), 1)
        mse_loss = torch.sum((D[mask] - predicted_distances[mask]) ** 2)

        if epoch % 1000 == 0 and verbose:
            print(f"[TrainDbar_Hybrid] Epoch {epoch}, MSE Loss: {mse_loss.item():.3e}")

        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

        # stopping criterion
        if epoch > 0 and abs(mse_loss.item() - prev_mse_loss) / prev_mse_loss < tol:
            if verbose:
                print(f"Converged at epoch {epoch}, MSE Loss: {mse_loss.item():.3e}")
            break
        prev_mse_loss = mse_loss.item()


def trainY(
    model,
    X,
    Y,
    device,
    lr=1e-3,
    weight_decay=1e-5,
    beta=1.0,
    tol=1e-6,
    max_epochs=10000,
    batch_size=None,
    verbose=True,
):
    """
    Optimize cluster centers Y while keeping model fixed.
    Uses mini-batches of X if batch_size is specified.

    Args:
        model: Trained ADEN model (fixed).
        X: Tensor of data points, shape (N, input_dim).
        Y: Tensor of initial cluster centroids, shape (M, input_dim).
        device: torch.device.
        lr: Learning rate for Y optimization.
        weight_decay: Optimizer weight decay.
        beta: Inverse temperature parameter for free energy.
        tol: Relative tolerance for stopping criterion.
        max_epochs: Maximum optimization steps.
        batch_size: If not None, number of samples per batch for X.
        verbose: If True, print progress every 100 steps.

    Returns:
        y_opt: Optimized cluster centers, shape (M, input_dim).
        history: List of free energy values over iterations.
    """

    N = X.shape[0]

    # Clone Y and make it trainable
    y_opt = Y.clone().detach().to(device).float()
    y_opt.requires_grad_(True)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    optimizer_y = torch.optim.AdamW([y_opt], lr=lr, weight_decay=weight_decay)

    # Initialize free energy
    F_old = torch.tensor(float("inf"), device=device)
    history = []

    for epoch in range(max_epochs+1):
        # Shuffle X for batching
        perm = torch.randperm(N, device=device)
        F_epoch = 0.0

        for start in range(0, N, batch_size or N):
            end = min(start + (batch_size or N), N)
            X_batch = X[perm[start:end]].to(device).float()  # (B, input_dim)

            # Compute distances via fixed model

            d_s = model(X_batch.unsqueeze(0), y_opt.unsqueeze(0))[0]  # (B, M)

            d_mins = torch.min(d_s, dim=-1, keepdim=True).values  # (B, 1)
            # Free energy contribution for this batch
            F_batch = -(1.0 / beta) * torch.sum(
                torch.log(torch.sum(torch.exp(-beta * (d_s - d_mins)), dim=-1))
                - beta * d_mins.squeeze(-1)
            ) / N  # normalize by total N

            F_batch.backward(retain_graph=True if end < N else False)
            F_epoch += F_batch.item()

        optimizer_y.step()
        optimizer_y.zero_grad()

        history.append(F_epoch)

        # Logging
        if verbose and epoch % 1000 == 0:
            print(f"[trainY] Epoch {epoch}, F: {F_epoch:.3e}")

        # Convergence check
        if abs(F_epoch - F_old) / (abs(F_old) + 1e-8) < tol:
            if verbose:
                print(f"[trainY] Converged at epoch {epoch}, F: {F_epoch:.3e}")
            break

        F_old = torch.tensor(F_epoch, device=device)

    return y_opt.detach(), history


def TrainAnneal(
    model,
    X,
    Y,
    device,
    # TrainDbar hyperparameters
    epochs_dbar=1000,
    batch_size_dbar=32,
    num_samples_in_batch_dbar=128,
    lr_dbar=1e-4,
    weight_decay_dbar=1e-5,
    tol_train_dbar=1e-6,
    gamma_dbar=1000.0,
    probs_dbar=None,
    # trainY hyperparameters
    epochs_train_y=10000,
    batch_size_train_y=None,
    lr_train_y=1e-3,
    weight_decay_train_y=1e-5,
    tol_train_y=1e-6,
    # annealing schedule
    beta_init=1e-3,
    beta_final=10.0,
    beta_growth_rate=10.0,
    perturbation_std=0.01,
):
    """
    Run the annealing loop alternating TrainDbar and trainY.

    Args:
        model: ADEN model.
        X: Tensor of data points, shape (N, input_dim).
        Y: Tensor of initial cluster centers, shape (M, input_dim).
        device: torch.device.
        epochs_dbar, batch_size_dbar, ... : hyperparameters for TrainDbar.
        epochs_train_y, batch_size_train_y, ... : hyperparameters for trainY.
        beta_init, beta_final, beta_growth_rate: annealing schedule.
        perturbation_std: std of Gaussian noise added to Y each iteration.
        probs_dbar: optional (M, M, N) probability tensor for TrainDbar.
    Returns:
        Y_final: optimized cluster centers
        history_y: list of free energy histories from each trainY call
    """

    M, input_dim = Y.shape
    beta = beta_init
    history_y_all = []
    history_pi_all = []
    while beta <= beta_final:
        print(f"\n=== Annealing step: Beta = {beta:.4f} ===")

        # Perturb Y
        Y = Y + torch.randn(M, input_dim, device=device) * perturbation_std

        # --- TrainDbar ---
        TrainDbar_Hybrid(
            model,
            X,
            Y,
            device,
            epochs=epochs_dbar,
            batch_size=batch_size_dbar,
            num_samples_in_batch=num_samples_in_batch_dbar,
            lr=lr_dbar,
            weight_decay=weight_decay_dbar,
            tol=tol_train_dbar,
            gamma=gamma_dbar,
            probs=probs_dbar,
            perturbation_std=perturbation_std,
            verbose=True
        )

        # --- trainY ---
        Y, history_y = trainY(
            model,
            X,
            Y,
            device,
            lr=lr_train_y,
            weight_decay=weight_decay_train_y,
            beta=beta,
            tol=tol_train_y,
            max_epochs=epochs_train_y,
            batch_size=batch_size_train_y,
            verbose=True,
        )
        with torch.no_grad():
            pi = (
                torch.argmin(model(X.unsqueeze(0), Y.unsqueeze(0))[0], dim=-1).cpu().numpy()
            )  # (N, M)
        history_y_all.append(Y.clone().detach().cpu().numpy())
        history_pi_all.append(pi)
        # Increase beta
        beta *= beta_growth_rate
        # model.reset_weights()  # Reset model weights for each temperature

    return Y, history_y_all, history_pi_all
