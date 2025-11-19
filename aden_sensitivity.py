import argparse
import itertools
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ADEN import ADEN
from ADENTrain import TrainAnneal
from Env import ClusteringEnvTorch, ClusteringEnvNumpy
from TestCaseGenerator import data_RLClustering
import utils
from ClusteringGroundTruth import distortion


def build_model(d, cfg):
    return ADEN(
        input_dim=d,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        device=cfg["device"],
    )


def init_centers(X, M, d, method: str, device: torch.device):
    if method == "mean_noise":
        return torch.mean(X, dim=0, keepdim=True).to(device) + 0.01 * torch.randn(M, d, device=device)
    elif method == "sample":
        idx = torch.randperm(X.shape[0], device=device)[:M]
        return X[idx].clone()
    else:
        raise ValueError(f"Unknown init method: {method}")


def run_trial(
    X_np,
    env_cfg,
    model_cfg,
    train_cfg,
    anneal_cfg,
    seed: int,
    run_root: str,
    print_size: int,
):
    device = model_cfg["device"]
    utils.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # tensors
    X = torch.tensor(X_np).float().to(device)
    N, d = X.shape
    M = model_cfg.get("M")
    if M is None:
        # default to the number of clusters from data_RLClustering
        # assume data_RLClustering returns M as 2nd output; we passed it in main
        raise ValueError("Model cfg must include M (number of clusters)")

    # init Y
    Y = init_centers(X, M, d, train_cfg["init_method"], device)

    # environment
    env = ClusteringEnvTorch(
        n_data=N,
        n_clusters=M,
        n_features=d,
        parametrized=True,
        kappa=env_cfg["kappa"],
        gamma=env_cfg["gamma"],
        zeta=env_cfg["zeta"],
        T=env_cfg["T"],
        T_p=torch.tensor(env_cfg["T_P"]).to(device) if env_cfg.get("T_P") is not None else None,
        device=device,
    )

    # writer
    run_name = (
        f"IDX4_M{M}_"
        f"kappa{env_cfg['kappa']}gam{env_cfg['gamma']}zet{env_cfg['zeta']}T{env_cfg['T']}_"
        f"seed{seed}_"
        f"D{model_cfg['d_model']}_L{model_cfg['n_layers']}_H{model_cfg['n_heads']}_FF{model_cfg['d_ff']}_DO{model_cfg['dropout']}_"
        f"EpD{train_cfg['epochs_dbar']}BSD{train_cfg['batch_size_dbar']}NSD{train_cfg['num_samples_in_batch_dbar']}LRD{train_cfg['lr_dbar']}_"
        f"EpY{train_cfg['epochs_train_y']}LRY{train_cfg['lr_train_y']}_"
        f"{anneal_cfg['beta_init']}to{anneal_cfg['beta_final']}rate{anneal_cfg['beta_growth_rate']}_Pert{anneal_cfg['perturbation_std']}"
    )
    log_dir = os.path.join(run_root, run_name)
    writer = SummaryWriter(log_dir=log_dir)

    # train
    Y_opt, pi_opt, history_y_all, history_pi_all, Betas = TrainAnneal(
        model=build_model(d, model_cfg),
        X=X,
        Y=Y,
        env=env,
        device=device,
        # dbar
        epochs_dbar=train_cfg["epochs_dbar"],
        batch_size_dbar=train_cfg["batch_size_dbar"],
        num_samples_in_batch_dbar=train_cfg["num_samples_in_batch_dbar"],
        lr_dbar=train_cfg["lr_dbar"],
        weight_decay_dbar=train_cfg["weight_decay_dbar"],
        tol_train_dbar=train_cfg["tol_train_dbar"],
        # y
        epochs_train_y=train_cfg["epochs_train_y"],
        batch_size_train_y=train_cfg["batch_size_train_y"],
        lr_train_y=train_cfg["lr_train_y"],
        weight_decay_train_y=train_cfg["weight_decay_train_y"],
        tol_train_y=train_cfg["tol_train_y"],
        # anneal
        beta_init=anneal_cfg["beta_init"],
        beta_final=anneal_cfg["beta_final"],
        beta_growth_rate=anneal_cfg["beta_growth_rate"],
        perturbation_std=anneal_cfg["perturbation_std"],
        # logging
        writer=writer,
        print_size=print_size,
    )

    # compute distortion in numpy env for final metrics
    try:
        X_final = X.detach().cpu().numpy()
        Y_final = Y_opt.detach().cpu().numpy()
    except Exception:
        X_final = X_np
        Y_final = np.array(Y_opt)

    env_np = ClusteringEnvNumpy(
        n_data=X_final.shape[0],
        n_clusters=Y_final.shape[0],
        n_features=X_final.shape[1],
        parametrized=True,
        kappa=env_cfg["kappa"],
        gamma=env_cfg["gamma"],
        zeta=env_cfg["zeta"],
        T=env_cfg["T"],
        T_p=None,
    )
    rho = np.ones(X_final.shape[0]) / X_final.shape[0]
    dist_val = distortion(X_final, Y_final, rho, pi_opt, env_np)
    writer.add_scalar("metrics/final_distortion", float(dist_val))

    writer.add_text("config/env", str(env_cfg))
    writer.add_text("config/model", str({k: v for k, v in model_cfg.items() if k != 'device'}))
    writer.add_text("config/train", str(train_cfg))
    writer.add_text("config/anneal", str(anneal_cfg))
    writer.flush()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="ADEN sensitivity study over init and hyperparameters")
    parser.add_argument("--runs_root", type=str, default="runs/sensitivity/", help="TensorBoard log root")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1], help="Random seeds to try")
    parser.add_argument("--device", type=str, default="auto", help="cuda|cpu|auto")
    parser.add_argument("--print_size", type=int, default=10, help="Logging step interval")
    args = parser.parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    os.makedirs(args.runs_root, exist_ok=True)

    # data
    X_np, M, T_P, N, d = data_RLClustering(4)

    # environment grid (small by default)
    env_param_grid = [
        {"kappa": 0.2, "gamma": 0.0, "zeta": 1.0, "T": 0.1, "T_P": T_P},
        {"kappa": 0.5, "gamma": 0.0, "zeta": 1.0, "T": 0.1, "T_P": T_P},
    ]

    # model configs to sweep
    model_cfg_grid = [
        {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff": 128, "dropout": 0.01},
        {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff": 256, "dropout": 0.01},
    ]

    # training configs to sweep
    train_cfg_grid = [
        {
            # dbar
            "epochs_dbar": 5000,
            "batch_size_dbar": 32,
            "num_samples_in_batch_dbar": 128,
            "lr_dbar": 1e-4,
            "weight_decay_dbar": 1e-5,
            "tol_train_dbar": 1e-6,
            # y
            "epochs_train_y": 1000,
            "batch_size_train_y": None,
            "lr_train_y": 1e-4,
            "weight_decay_train_y": 1e-5,
            "tol_train_y": 1e-6,
            # init
            "init_method": "mean_noise",
        },
        {
            # dbar
            "epochs_dbar": 5000,
            "batch_size_dbar": 32,
            "num_samples_in_batch_dbar": 128,
            "lr_dbar": 1e-4,
            "weight_decay_dbar": 1e-5,
            "tol_train_dbar": 1e-6,
            # y
            "epochs_train_y": 1000,
            "batch_size_train_y": None,
            "lr_train_y": 1e-4,
            "weight_decay_train_y": 1e-5,
            "tol_train_y": 1e-6,
            # init
            "init_method": "sample",
        },
        {
            # dbar
            "epochs_dbar": 5000,
            "batch_size_dbar": 32,
            "num_samples_in_batch_dbar": 128,
            "lr_dbar": 5e-4,
            "weight_decay_dbar": 1e-5,
            "tol_train_dbar": 1e-6,
            # y
            "epochs_train_y": 1000,
            "batch_size_train_y": None,
            "lr_train_y": 5e-4,
            "weight_decay_train_y": 1e-5,
            "tol_train_y": 1e-6,
            # init
            "init_method": "mean_noise",
        },
        {
            # dbar
            "epochs_dbar": 5000,
            "batch_size_dbar": 32,
            "num_samples_in_batch_dbar": 128,
            "lr_dbar": 5e-4,
            "weight_decay_dbar": 1e-5,
            "tol_train_dbar": 1e-6,
            # y
            "epochs_train_y": 1000,
            "batch_size_train_y": None,
            "lr_train_y": 5e-4,
            "weight_decay_train_y": 1e-5,
            "tol_train_y": 1e-6,
            # init
            "init_method": "sample",
        },
    ]

    # annealing config (fixed defaults)
    anneal_cfg = {
        "beta_init": 10.0,
        "beta_final": 10000.0,
        "beta_growth_rate": 10.0,
        "perturbation_std": 0.01,
    }
    simulation_time = datetime.now().strftime("/%Y%m%d_%H%M%S")
    # run all combos
    for env_cfg in env_param_grid:
        for seed in args.seeds:
            for model_cfg, train_cfg in itertools.product(model_cfg_grid, train_cfg_grid):
                model_cfg_local = dict(model_cfg)
                model_cfg_local["device"] = device
                model_cfg_local["M"] = M
                run_trial(
                    X_np=X_np,
                    env_cfg=env_cfg,
                    model_cfg=model_cfg_local,
                    train_cfg=train_cfg,
                    anneal_cfg=anneal_cfg,
                    seed=seed,
                    run_root=args.runs_root + simulation_time,
                    print_size=args.print_size,
                )


if __name__ == "__main__":
    main()
