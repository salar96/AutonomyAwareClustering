import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from ADEN import ADEN
from TestCaseGenerator import data_RLClustering
from ADENTrain import TrainAnneal
import utils
from Env import ClusteringEnvNumpy, ClusteringEnvTorch
from ClusteringGroundTruth import cluster_gt, distortion
import pickle
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------------------------------------
# SCENARIOS (exactly those in the benchmark table)
# Each entry: (kappa, gamma, zeta, T)
SCENARIOS = [
    (0.1, 0.0, 1.0, 0.01),
    (0.1, 0.0, 1.0, 100.0),
    (0.1, 0.5, 1.0, 0.01),
    (0.2, 0.0, 1.0, 0.01),
    (0.2, 0.0, 1.0, 100.0),
    (0.2, 0.5, 1.0, 0.01),
    (0.2, 0.5, 1.0, 100.0),
    (0.3, 0.0, 1.0, 0.01),
    (0.3, 0.0, 1.0, 100.0),
    (0.3, 0.5, 1.0, 0.01),
    (0.3, 0.5, 1.0, 100.0),
    (0.4, 0.0, 1.0, 0.01),
    (0.4, 0.0, 1.0, 100.0),
    (0.4, 0.5, 1.0, 0.01),
    (0.4, 0.5, 1.0, 100.0),
    (0.5, 0.0, 1.0, 100.0),
    (0.5, 0.5, 1.0, 0.01),
    (0.5, 0.5, 1.0, 100.0),
]

N_RUNS = 1
parametrized = True

# ----------------------------------------------------------
# HYPERPARAMETERS
INPUT_DIM = None  # set after data loading
D_MODEL = 64
N_LAYERS = 4
N_HEADS = 8
D_FF = 128
DROPOUT = 0.01

EPOCHS_DBAR = 1000
BATCH_SIZE_DBAR = 32
NUM_SAMPLES_IN_BATCH_DBAR = 128
LR_DBAR = 1e-4
WEIGHT_DECAY_DBAR = 1e-5
TOL_TRAIN_DBAR = 1e-6

EPOCHS_TRAIN_Y = 100
BATCH_SIZE_TRAIN_Y = None
LR_TRAIN_Y = 1e-4
WEIGHT_DECAY_TRAIN_Y = 1e-5
TOL_TRAIN_Y = 1e-4

BETA_INIT = 10.0
BETA_F = 50000.0
BETA_GROWTH_RATE = 1.1
PERTURBATION_STD = 0.01

# ----------------------------------------------------------
# LOADING DATA (fixed across runs — randomness comes from Y init and model weights)
utils.set_seed(0)
X_base, M, T_P, N, d = data_RLClustering(4)
INPUT_DIM = d
rho = np.ones(N) / N

os.makedirs("Benchmark_new", exist_ok=True)

print("Scenarios:", len(SCENARIOS))
print("N_RUNS:", N_RUNS)
print("D_model:", D_MODEL, "N_layers:", N_LAYERS, "N_heads:", N_HEADS, "D_ff:", D_FF)

print(f"Model: ADEN | input_dim={INPUT_DIM} d_model={D_MODEL} n_layers={N_LAYERS} n_heads={N_HEADS} d_ff={D_FF}")

# ----------------------------------------------------------
# RESULTS ACCUMULATOR: scenario_key -> list of (error_opt, error_ig) per run
results_acc = defaultdict(lambda: {"error_opt": [], "error_ig": []})

# ----------------------------------------------------------
# MAIN LOOP
for run_id in range(N_RUNS):
    seed = run_id
    utils.set_seed(seed)
    print(f"\n\033[95m{'='*60}\033[0m")
    print(f"\033[95mRUN {run_id + 1} / {N_RUNS}  (seed={seed})\033[0m")
    print(f"\033[95m{'='*60}\033[0m\n")

    # Re-create model with default Kaiming init (matches original behavior for scenario 1)
    model = ADEN(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        device=device,
    )

    # Fresh data tensors with this run's random centroid init
    X = torch.tensor(X_base).float().to(device)
    Y_init = torch.mean(X, dim=0, keepdim=True).to(device) + 0.01 * torch.randn(M, d).to(device)
    X_np = X.cpu().numpy()
    Y_np = Y_init.cpu().numpy()

    # --- Baseline: autonomy ignored ---
    env_ig = ClusteringEnvNumpy(
        n_data=N, n_clusters=M, n_features=d,
        parametrized=False,
        kappa=None, gamma=None, zeta=None, T=None, T_p=None,
    )
    Y_ig, pi_ig, _, _, _ = cluster_gt(
        X_np, Y_np, rho, env_ig,
        beta_min=BETA_INIT, beta_max=BETA_F, tau=BETA_GROWTH_RATE,
    )
    print("\033[94mBaseline (autonomy ignored) obtained.\033[0m")

    # --- Per-scenario loop ---
    for kappa, gamma, zeta, T in SCENARIOS:
        scenario_key = (kappa, gamma, zeta, T)
        scenario_name = f"Benchmark_parametrized{parametrized}_kappa{kappa}_gamma{gamma}_zeta{zeta}_T{T}"
        print(f"\033[93mRun {run_id+1} | Scenario: {scenario_name}\033[0m")

        # Ground truth
        env_np = ClusteringEnvNumpy(
            n_data=N, n_clusters=M, n_features=d,
            parametrized=parametrized,
            kappa=kappa, gamma=gamma, zeta=zeta, T=T, T_p=T_P,
        )
        Y_GT, pi_GT, _, _, _ = cluster_gt(
            X_np, Y_np, rho, env_np,
            beta_min=BETA_INIT, beta_max=BETA_F, tau=BETA_GROWTH_RATE,
        )
        print("\033[92mGround truth obtained.\033[0m")

        # ADEN training
        env_torch = ClusteringEnvTorch(
            n_data=N, n_clusters=M, n_features=d,
            parametrized=parametrized,
            kappa=kappa, gamma=gamma, zeta=zeta, T=T,
            T_p=torch.tensor(T_P), device=device,
        )
        Y_opt, pi_opt, _, _, _ = TrainAnneal(
            model, X, Y_init.clone(), env_torch, device,
            epochs_dbar=EPOCHS_DBAR,
            batch_size_dbar=BATCH_SIZE_DBAR,
            num_samples_in_batch_dbar=NUM_SAMPLES_IN_BATCH_DBAR,
            lr_dbar=LR_DBAR,
            weight_decay_dbar=WEIGHT_DECAY_DBAR,
            tol_train_dbar=TOL_TRAIN_DBAR,
            epochs_train_y=EPOCHS_TRAIN_Y,
            batch_size_train_y=BATCH_SIZE_TRAIN_Y,
            lr_train_y=LR_TRAIN_Y,
            weight_decay_train_y=WEIGHT_DECAY_TRAIN_Y,
            tol_train_y=TOL_TRAIN_Y,
            beta_init=BETA_INIT,
            beta_final=BETA_F,
            beta_growth_rate=BETA_GROWTH_RATE,
            perturbation_std=PERTURBATION_STD,
        )
        print("\033[92mADEN training completed.\033[0m")

        # D-gap computation
        dist_gt = distortion(X_np, Y_GT, rho, pi_GT, env_np)
        dist_opt = distortion(X_np, Y_opt.cpu().numpy(), rho, pi_opt, env_np)
        dist_ig = distortion(X_np, Y_ig, rho, pi_ig, env_np)
        error_opt = (dist_opt - dist_gt) / dist_gt * 100.0
        error_ig = (dist_ig - dist_gt) / dist_gt * 100.0
        results_acc[scenario_key]["error_opt"].append(error_opt)
        results_acc[scenario_key]["error_ig"].append(error_ig)
        print(f"  D-gap opt: {error_opt:.2f}%  |  D-gap ig: {error_ig:.2f}%")

        # Save per-run pkl
        save_dict = {
            "scenario_name": scenario_name,
            "run_id": run_id,
            "seed": seed,
            "Y_GT": Y_GT,
            "pi_GT": pi_GT,
            "Y_opt": Y_opt.cpu().numpy(),
            "pi_opt": pi_opt,
            "Y_ig": Y_ig,
            "pi_ig": pi_ig,
            "error_opt": error_opt,
            "error_ig": error_ig,
        }
        fname = f"Benchmark_new/{scenario_name}_run{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"  Saved: {fname}")
        # Reset to Xavier init for subsequent scenarios (matches original behavior)
        model.reset_weights()
        print()

print("\033[95mAll runs completed. Aggregating results...\033[0m\n")

# ----------------------------------------------------------
# AGGREGATE AND BUILD TABLE
rows = []
for kappa, gamma, zeta, T in SCENARIOS:
    key = (kappa, gamma, zeta, T)
    vals = results_acc[key]
    opt_arr = np.array(vals["error_opt"])
    ig_arr = np.array(vals["error_ig"])
    rows.append({
        "kappa": kappa,
        "gamma": gamma,
        "zeta": zeta,
        "T": T,
        "error_opt_mean": np.mean(opt_arr),
        "error_opt_std": np.std(opt_arr, ddof=1) if len(opt_arr) > 1 else 0.0,
        "error_ig_mean": np.mean(ig_arr),
        "error_ig_std": np.std(ig_arr, ddof=1) if len(ig_arr) > 1 else 0.0,
    })

summary_df = pd.DataFrame(rows)
pd.set_option("display.precision", 2)
print(summary_df.to_string(index=False))

# ----------------------------------------------------------
# SAVE SUMMARY PKL
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_pkl_path = f"Benchmark_new/summary_{timestamp}.pkl"
with open(summary_pkl_path, "wb") as f:
    pickle.dump({"summary": rows, "results_acc": dict(results_acc)}, f)
print(f"\nSummary pkl saved: {summary_pkl_path}")

# ----------------------------------------------------------
# SAVE LATEX TABLE
def fmt(mean, std):
    return f"{mean:.2f} $\\pm$ {std:.2f}"

n_rows = len(rows)
half = n_rows // 2

def build_tabular(subset):
    lines = []
    lines.append(r"\begin{tabular}{|c|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(
        r"$\kappa$ & $\bar{\gamma}$ & $\zeta$ & $T$ "
        r"& \makecell{$D$-Gap (\%) \\ Algorithm~\ref{Alg: AutonomyUnknown}} "
        r"& \makecell{$D$-Gap (\%) \\ Autonomy \\Ignored} \\ \hline"
    )
    for r in subset:
        T_str = f"{r['T']:.2f}".rstrip("0").rstrip(".")
        line = (
            f"{r['kappa']} & {r['gamma']} & {int(r['zeta'])} & {T_str} "
            f"& {fmt(r['error_opt_mean'], r['error_opt_std'])} "
            f"& {fmt(r['error_ig_mean'], r['error_ig_std'])} \\\\"
        )
        lines.append(line)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)

tex_lines = []
tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(
    r"\caption{$D$-gap (\%) of Algorithm~\ref{Alg: AutonomyUnknown} and baseline that ignores autonomy, "
    r"both relative to Algorithm~\ref{Alg: AutonomyKnown}. Mean $\pm$ std over "
    + str(N_RUNS) + r" runs.}"
)
tex_lines.append(r"\label{tab: benchmark}")
tex_lines.append(r"\footnotesize")
tex_lines.append(r"\begin{subtable}[t]{0.5\textwidth}")
tex_lines.append(r"\centering")
tex_lines.append(build_tabular(rows[:half]))
tex_lines.append(r"\end{subtable}%")
tex_lines.append(r"\hfill")
tex_lines.append(r"\begin{subtable}[t]{0.5\textwidth}")
tex_lines.append(r"\centering")
tex_lines.append(build_tabular(rows[half:]))
tex_lines.append(r"\end{subtable}")
tex_lines.append(r"\end{table}")

tex_str = "\n".join(tex_lines)
tex_path = f"Benchmark_new/table_{timestamp}.tex"
with open(tex_path, "w") as f:
    f.write(tex_str)
print(f"LaTeX table saved: {tex_path}")
print("\nLaTeX table preview:\n")
print(tex_str)
