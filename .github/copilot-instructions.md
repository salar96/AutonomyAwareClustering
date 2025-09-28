# AutonomyAwareClustering AI Coding Instructions

This codebase implements **Autonomy-Aware Clustering** using reinforcement learning principles. The system combines neural networks (ADEN), Markov Decision Processes, and annealing optimization to learn adaptive distance metrics for clustering tasks.

## Architecture Overview

### Core Components
- **ADEN** (`ADEN.py`): Adaptive Distance Estimation Network with multi-head attention that learns context-aware distance functions
- **Environment** (`Env.py`): Clustering environments (NumPy/PyTorch) modeling transition probabilities p(k|j,i) between cluster assignments
- **Ground Truth** (`ClusteringGroundTruth.py`): Reference implementations using analytical solutions to clustering problems
- **Training** (`ADENTrain.py`): Two-phase training system with distance learning and centroid optimization

### Data Flow Pattern
1. **Data Generation**: `TestCaseGenerator.py` → synthetic clustering datasets
2. **Environment Setup**: `Env.py` → transition probability matrices based on parametrized distances
3. **Training Phase**: `ADENTrain.py` → alternating optimization (ADEN training + centroid updates)
4. **Benchmarking**: `benchmark*.py` → systematic comparison against ground truth solutions

## Key Conventions

### Environment Parameters
Critical hyperparameters defining clustering behavior:
- `kappa`: exploration probability (0.1-0.5 typical)
- `gamma`: weight for data-cluster distances d(i,k) 
- `zeta`: weight for cluster-cluster distances d(j,k)
- `T`: softmax temperature controlling transition randomness
- `parametrized`: boolean for distance-based vs fixed transitions

### Training Workflow
Two-phase annealing schedule in `TrainAnneal()`:
1. **TrainDbar**: Train ADEN on expected distances using Monte Carlo sampling
2. **TrainY**: Optimize cluster centroids via gradient descent on free energy
3. **Beta Annealing**: Increase β (inverse temperature) to sharpen cluster assignments

### File Naming Patterns
- Results saved with descriptive names: `Benchmark_parametrized{bool}_kappa{val}_gamma{val}_zeta{val}_T{val}_{timestamp}.pkl`
- Animations: `Clustering_GT_N{samples}_M{clusters}_d{dims}_betaMin{val}_betaMax{val}_tau{rate}_kappa{val}_gamma{val}_zeta{val}_T{val}.gif`

## Development Workflows

### Running Benchmarks
```python
# Full benchmark suite
python benchmark.py  # Creates results in Benchmark/ directory

# Single scenario testing  
python benchmark_UDT.py  # Focused on specific parameter combinations
```

### Training ADEN Models
```python
from ADENTrain import TrainAnneal
from ADEN import ADEN
from Env import ClusteringEnvTorch

# Standard architecture
model = ADEN(input_dim=d, d_model=64, n_layers=4, n_heads=8, d_ff=128, dropout=0.01)

# Training with annealing
Y_opt, pi_opt, _, _, _ = TrainAnneal(
    model, X, Y, env, device,
    epochs_dbar=1000, epochs_train_y=100,
    beta_init=10.0, beta_final=10000.0, beta_growth_rate=1.1
)
```

### Notebook Execution
- `DeepClusteringParametrized.ipynb`: Main training experiments with detailed hyperparameter tracking
- `TabularRL_Clustering.ipynb`: Classical RL approach for comparison
- Always set `utils.set_seed(0)` for reproducibility
- Use `%env CUDA_LAUNCH_BLOCKING=1` for debugging CUDA issues

## Critical Implementation Details

### Distance Computation
ADEN enhances base squared Euclidean distances:
```python
base_distances = torch.sum((data_expanded - centers_expanded)**2, dim=-1)
adaptive_distances = base_distances + temperature * learned_deviations
```

### Transition Probabilities
Environment computes p(k|j,i) based on utility functions:
```python
u_k(j,i) = zeta * d(j,k) + gamma * d(i,k)  # cluster-cluster + data-cluster distances
p(k|j,i) = kappa * softmax(-u/T) if k≠j else (1-kappa)  # exploration vs exploitation
```

### Annealing Schedule
Critical for convergence - gradually increase β to sharpen assignments:
```python
beta *= tau  # multiplicative growth (typically tau=1.1)
Y += perturbation_std * randn()  # add noise to escape local minima
```

## Integration Points

### GPU/CPU Compatibility
- PyTorch environments (`ClusteringEnvTorch`) for GPU training
- NumPy environments (`ClusteringEnvNumpy`) for ground truth computation
- Always specify `device` parameter in model instantiation

### Result Storage
- Pickle files contain: `{Y_GT, pi_GT, Y_opt, pi_opt, Y_ig, pi_ig}` for comparison
- Use `utils.Chamfer_dist()` and `utils.Hungarian_dist()` for clustering quality metrics
- Timestamp-based filenames prevent overwriting results

### Visualization
- `animator.py`: Generate GIF animations of clustering evolution
- `Plotter.py`: Static visualizations of results
- Results stored in `animations/` directory with descriptive filenames

## Performance Considerations

- Batch processing in ADEN training: `batch_size=32`, `num_samples_in_batch=128`
- Memory-efficient sampling in environments using vectorized operations
- Model weight reset (`model.reset_weights()`) between benchmark scenarios
- Early stopping based on tolerance thresholds (`tol_train_dbar=1e-6`)

When modifying this codebase, maintain the two-phase training paradigm and preserve the annealing schedule structure. The environment parameters directly control clustering behavior - small changes can significantly impact results.