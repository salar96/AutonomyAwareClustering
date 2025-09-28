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
BETA_F = 50000.0
BETA_GROWTH_RATE = 1.1
PERTURBATION_STD = 0.01

parametrized = True
eps_list = [0.1, 0.2, 0.3, 0.4, 0.5]
gamma_list = [0.0, 0.5]
zeta_list = [1.0, 1.0]
T_list = [100.0, 0.01]

rho = np.ones(N) / N
print("hyperparameters used are:")
print("parametrized:", parametrized)
print("eps_list:", eps_list)
print("gamma_list:", gamma_list)
print("zeta_list:", zeta_list)
print("T_list:", T_list)
print("D_model:", D_MODEL)
print("N_layers:", N_LAYERS)
print("N_heads:", N_HEADS)
print("D_ff:", D_FF)
print("dropout:", DROPOUT)
print("EPOCHS_DBAR:", EPOCHS_DBAR)
print("BATCH_SIZE_DBAR:", BATCH_SIZE_DBAR)
print("NUM_SAMPLES_IN_BATCH_DBAR:", NUM_SAMPLES_IN_BATCH_DBAR)
print("LR_DBAR:", LR_DBAR)
print("WEIGHT_DECAY_DBAR:", WEIGHT_DECAY_DBAR)
print("TOL_TRAIN_DBAR:", TOL_TRAIN_DBAR)
print("EPOCHS_TRAIN_Y:", EPOCHS_TRAIN_Y)
print("BATCH_SIZE_TRAIN_Y:", BATCH_SIZE_TRAIN_Y)
print("LR_TRAIN_Y:", LR_TRAIN_Y)
print("WEIGHT_DECAY_TRAIN_Y:", WEIGHT_DECAY_TRAIN_Y)
print("TOL_TRAIN_Y:", TOL_TRAIN_Y)
print("BETA_INIT:", BETA_INIT)
print("BETA_F:", BETA_F)
print("BETA_GROWTH_RATE:", BETA_GROWTH_RATE)
print("PERTURBATION_STD:", PERTURBATION_STD)

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
# First obtain Y locations if transition probabilities were completely ignored
env_ig = ClusteringEnvNumpy(
    n_data=N,
    n_clusters=M,
    n_features=d,
    parametrized=False,
    eps=None,
    gamma=None,
    zeta=None,
    T=None,
    T_p=None,
)

Y_ig, pi_ig, _, _, _ = cluster_gt(
    X_np,
    Y_np,
    rho,
    env_ig,
    beta_min=BETA_INIT,
    beta_max=BETA_F,
    tau=BETA_GROWTH_RATE,
)
print("\033[94mResults without transition probabilities obtained.\033[0m")
# LOOPING OVER SCENARIOS
for eps in eps_list:
    for idx, gamma in enumerate(gamma_list):
        zeta = zeta_list[idx]  # pairing zeta with gamma
        for T in T_list:
            scenario_name = f"Benchmark_parametrized{parametrized}_eps{eps}_gamma{gamma}_zeta{zeta}_T{T}"
            print("\033[93mScenario:", scenario_name, "\033[0m")
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

            Y_GT, pi_GT, _, _, _ = cluster_gt(
                X_np,
                Y_np,
                rho,
                env_np,
                beta_min=BETA_INIT,
                beta_max=BETA_F,
                tau=BETA_GROWTH_RATE,
            )
            print("\033[92mGround truth obtained.\033[0m")
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
            print("\033[92mADEN training completed.\033[0m")
            # SAVING RESULTS of ground truth and ADEN
            save_dict = {
                "scenario_name": scenario_name,
                "Y_GT": Y_GT,
                "pi_GT": pi_GT,
                "Y_opt": Y_opt.cpu().numpy(),
                "pi_opt": pi_opt,
                "Y_ig": Y_ig,
                "pi_ig": pi_ig,
            }
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"Benchmark/{scenario_name}_{timestamp}.pkl", "wb") as f:
                pickle.dump(save_dict, f)
            print("Results saved.\n")
            # RESETTING MODEL
            model.reset_weights()
            print("\033[91mModel weights reset.\033[0m\n")
# ----------------------------------------------------------
print("All scenarios completed.")
# ----------------------------------------------------------
# Note: To run this benchmark, ensure that the "Benchmark" directory exists in the current working directory.
# The benchmark results will be saved as pickle files in that directory.
