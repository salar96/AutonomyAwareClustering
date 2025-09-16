import numpy as np
import torch
from ADEN import ADEN
from torchinfo import summary
from TestCaseGenerator import data_RLClustering
from ADENTrain import TrainAnneal
import utils
from Env import ClusteringEnvNumpy, ClusteringEnvTorch
from ClusteringGroundTruth import cluster_gt
import pickle
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
utils.set_seed(0)  # for reproducibility

# LOADING DATA
X, M, T_P, N, d = data_RLClustering(4)
X = torch.tensor(X).float().to(device)
Y = torch.mean(X, dim=0, keepdim=True).to(device) + 0.01 * torch.randn(M, d).to(device)
X_np = X.cpu().numpy()
Y_np = Y.cpu().numpy()
# ----------------------------------------------------------
# HYPERPARAMETERS
INPUT_DIM = d  # dimensionality of the input space
D_MODEL = 64  # dimensionality of the model
N_LAYERS = 4  # number of layers
N_HEADS = 8  # number of attention heads
D_FF = 128  # dimensionality of the feedforward network
DROPOUT = 0.01  # dropout rate

EPOCHS_DBAR = 1000
BATCH_SIZE_DBAR = 32
NUM_SAMPLES_IN_BATCH_DBAR = 128
LR_DBAR = 1e-4
WEIGHT_DECAY_DBAR = 1e-5
TOL_TRAIN_DBAR = 1e-6
PROBS_DBAR = torch.tensor(T_P)

EPOCHS_TRAIN_Y = 100
BATCH_SIZE_TRAIN_Y = None
LR_TRAIN_Y = 1e-4
WEIGHT_DECAY_TRAIN_Y = 1e-5
TOL_TRAIN_Y = 1e-4

BETA_INIT = 10.0
BETA_F = 100000.0
BETA_GROWTH_RATE = 1.1
PERTURBATION_STD = 0.01

parametrized = True
eps_list = [0.1, 0.3, 0.5, 0.7, 0.9]
gamma_list = [0.0, 0.5]
zeta_list = [0.5, 1.0]
T_list = [10, 1, 0.1, 0.01, 0.001]
# ----------------------------------------------------------
# MODEL INITIALIZATION
model = ADEN(
    input_dim=INPUT_DIM,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF,
    dropout=DROPOUT,
    device=device,
)

print(summary(model))
# ----------------------------------------------------------
# LOOPING OVER SCENARIOS
for eps in eps_list:
    for gamma in gamma_list:
        for zeta in zeta_list:
            for T in T_list:
                scenario_name = f"Benchmark_N{N}_M{M}_d{d}_betaMin{BETA_INIT}_betaMax{BETA_F}_tau{BETA_GROWTH_RATE}_eps{eps}_gamma{gamma}_zeta{zeta}_T{T}"
                print("Scenario:", scenario_name)
                # FIRST GETTING GROUND TRUTH
                env_np = ClusteringEnvNumpy(
                    n_data=N,
                    n_clusters=M,
                    n_features=d,
                    parametrized=parametrized,
                    eps=eps,
                    gamma=gamma,
                    zeta=zeta,
                    T=T,
                    T_p=T_P,
                )
                rho = np.ones(N) / N
                Y_GT, pi_GT, _, _, _ = cluster_gt(
                    X_np,
                    Y_np,
                    rho,
                    env_np,
                    beta_min=BETA_INIT,
                    beta_max=BETA_F,
                    tau=BETA_GROWTH_RATE,
                )
                print("Ground truth obtained.")
                # THEN TRAINING ADEN
                env_torch = ClusteringEnvTorch(
                    n_data=N,
                    n_clusters=M,
                    n_features=d,
                    parametrized=parametrized,
                    eps=eps,
                    gamma=gamma,
                    zeta=zeta,
                    T=T,
                    T_p=torch.tensor(T_P),
                    device=device,
                )
                Y_opt, pi_opt, _, _, _ = TrainAnneal(
                    model,
                    X,
                    Y.clone(),
                    env_torch,
                    device,
                    # TrainDbar hyperparameters
                    epochs_dbar=EPOCHS_DBAR,
                    batch_size_dbar=BATCH_SIZE_DBAR,
                    num_samples_in_batch_dbar=NUM_SAMPLES_IN_BATCH_DBAR,
                    lr_dbar=LR_DBAR,
                    weight_decay_dbar=WEIGHT_DECAY_DBAR,
                    tol_train_dbar=TOL_TRAIN_DBAR,
                    # trainY hyperparameters
                    epochs_train_y=EPOCHS_TRAIN_Y,
                    batch_size_train_y=BATCH_SIZE_TRAIN_Y,
                    lr_train_y=LR_TRAIN_Y,
                    weight_decay_train_y=WEIGHT_DECAY_TRAIN_Y,
                    tol_train_y=TOL_TRAIN_Y,
                    # annealing schedule
                    beta_init=BETA_INIT,
                    beta_final=BETA_F,
                    beta_growth_rate=BETA_GROWTH_RATE,
                    perturbation_std=PERTURBATION_STD,
                )
                print("ADEN training completed.")
                # SAVING RESULTS of ground truth and ADEN
                save_dict = {
                    "scenario_name": scenario_name,
                    "Y_GT": Y_GT,
                    "pi_GT": pi_GT,
                    "Y_opt": Y_opt.cpu().numpy(),
                    "pi_opt": pi_opt.cpu().numpy(),
                }
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"Benchmark/{scenario_name}_{timestamp}.pkl", "wb") as f:
                    pickle.dump(save_dict, f)
                print("Results saved.\n")
                # RESETTING MODEL
                model.reset_weights()
                print("Model weights reset.\n")
# ----------------------------------------------------------
print("All scenarios completed.")
# ----------------------------------------------------------
# Note: To run this benchmark, ensure that the "Benchmark" directory exists in the current working directory.
# The benchmark results will be saved as pickle files in that directory.

