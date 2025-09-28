# Autonomy-Aware Clustering

A novel approach to clustering that combines **deep learning**, **reinforcement learning**, and **Markov Decision Processes** to learn adaptive distance metrics for improved clustering performance under local autonomy. The system uses an Adaptive Distance Estimation Network (ADEN) that learns context-aware distance functions through interaction with parametrized clustering environments.

## 🎯 Key Features

- **Adaptive Distance Learning**: Neural network learns optimal distance metrics rather than using fixed Euclidean distances
- **Reinforcement Learning Framework**: Models clustering as a Markov Decision Process with transition probabilities between cluster assignments
- **Annealing Optimization**: Two-phase training with β-annealing for progressive refinement
- **GPU Acceleration**: Full CUDA support for large-scale clustering tasks
- **Comprehensive Benchmarking**: Systematic comparison against analytical ground truth solutions

![Phase Transition Animation](Phase_Transition.gif)

*Example: Phase transition behavior during β-annealing showing cluster formation and refinement*

## 🏗️ Architecture Overview

### Core Components

1. **ADEN** (`ADEN.py`): Adaptive Distance Estimation Network
   - Multi-head attention mechanism for learning context-aware distances
   - Combines base Euclidean distances with learned adaptive deviations
   - Temperature-scaled distance predictions with ReLU activation

2. **Clustering Environments** (`Env.py`): 
   - `ClusteringEnvNumpy`: CPU-based environment for ground truth computation
   - `ClusteringEnvTorch`: GPU-accelerated environment for neural network training
   - Parametrized transition probabilities p(k|j,i) based on utility functions

3. **Training System** (`ADENTrain.py`):
   - **TrainDbar**: Neural network training on expected distances via Monte Carlo sampling
   - **TrainY**: Cluster centroid optimization using gradient descent on free energy
   - **TrainAnneal**: Coordinated annealing schedule with β parameter growth

4. **Ground Truth Solver** (`ClusteringGroundTruth.py`):
   - Analytical solutions for clustering optimization when local autonomy is known
   - Reference implementations for benchmarking
   - Free energy minimization with scipy optimization

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch with CUDA support

### Setup
```bash
git clone https://github.com/salar96/AutonomyAwareClustering.git
cd AutonomyAwareClustering

# Install dependencies
pip install -r requirements.txt
```

### Data Requirements
The system includes synthetic data generators and supports real datasets:
- `TestCaseGenerator.py`: Multiple synthetic clustering scenarios
- `UTD19_London.mat`: Real-world sensor location data (included)
- Custom datasets via CSV import

## 🚀 Quick Start

### Basic Usage

```python
import torch
import numpy as np
from ADEN import ADEN
from Env import ClusteringEnvTorch
from ADENTrain import TrainAnneal
from TestCaseGenerator import data_RLClustering
import utils

# Set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.set_seed(0)

# Load synthetic dataset
X, M, T_P, N, d = data_RLClustering(4)  # 4-cluster 2D dataset
X = torch.tensor(X).float().to(device)
Y = torch.mean(X, dim=0, keepdim=True).to(device) + 0.01 * torch.randn(M, d).to(device)

# Create parametrized environment
env = ClusteringEnvTorch(
    n_data=N, n_clusters=M, n_features=d,
    parametrized=True, eps=0.4, gamma=0.0, zeta=1.0, T=0.01,
    device=device
)

# Initialize ADEN model
model = ADEN(input_dim=d, d_model=64, n_layers=4, n_heads=8, d_ff=128, dropout=0.01)

# Train with annealing
Y_opt, pi_opt, _, _, _ = TrainAnneal(
    model, X, Y, env, device,
    epochs_dbar=1000, epochs_train_y=100,
    beta_init=10.0, beta_final=10000.0, beta_growth_rate=1.1
)
```

### Running Benchmarks

```bash
# Full benchmark suite across parameter combinations
python benchmark.py

# Single scenario focused testing
python benchmark_UDT.py

# Results saved to Benchmark/ directory with timestamps
```

### Interactive Experiments

Use the provided Jupyter notebooks for experimentation:

```bash
# Main training notebook with synthetic and real-world data
jupyter notebook DeepClusteringParametrized.ipynb

# Classical RL comparison
jupyter notebook TabularRL_Clustering.ipynb

# Ground truth analysis
jupyter notebook Clustering_GT.ipynb
```

## 📊 Key Parameters

### Environment Parameters (Critical for Performance)
- `eps`: Exploration probability (0.1-0.5) - controls transition randomness
- `gamma`: Weight for data-cluster distances d(i,k) 
- `zeta`: Weight for cluster-cluster distances d(j,k)
- `T`: Softmax temperature - lower values = sharper transitions
- `parametrized`: Boolean - use distance-based vs fixed transition probabilities

### Model Architecture
- `d_model`: Internal embedding dimension (default: 64)
- `n_layers`: Number of attention blocks (default: 4)  
- `n_heads`: Multi-head attention heads (default: 8)
- `d_ff`: Feed-forward network dimension (default: 128)

### Training Hyperparameters
- `epochs_dbar`: ADEN training epochs per annealing step (1000-2000)
- `epochs_train_y`: Centroid optimization epochs per step (100)
- `beta_init/beta_final`: Annealing schedule bounds (10.0 to 10000.0)
- `beta_growth_rate`: Multiplicative growth factor (1.1)

## 🧮 Mathematical Framework

### Transition Probabilities
The environment computes cluster transition probabilities:

$u_k(j,i) = ζ·d(j,k) + γ·d(i,k)$

$p(k|j,i) = ε·softmax(-u_k/T) \quad if \quad  k≠j, \quad else \ (1-ε)$

### Analytical Ground Truth
The ground truth optimal assignments and centroids are given by:

$\pi_{Y}^{\beta}(j|i) 
= \mathrm{softmax}_j\!\big(-\beta\, d_{\mathrm{avg}}(x_i,y_j)\big) 
= \frac{\exp\{-\beta\, d_{\mathrm{avg}}(x_i,y_j)\}}{\sum_{\ell=1}^K \exp\{-\beta\, d_{\mathrm{avg}}(x_i,y_\ell)\}},$

$y_{\ell} 
= \frac{\sum_{i=1}^N \sum_{j=1}^K \rho(i)\, p(\ell|j,i)\, \pi_{Y}^{\beta}(j|i)\, x_i}
       {\sum_{i=1}^N \sum_{j=1}^K \rho(i)\, p(\ell|j,i)\, \pi_{Y}^{\beta}(j|i)},
\quad \forall~1 \leq \ell \leq K.$
### Adaptive Distance Function
ADEN enhances base distances with learned components:

$d_{adaptive}(i,k) = ||x_i - y_k||² + τ\bar{d}(x_i, y_k)$

where $\bar{d}(x_i, y_k)$ is the output of the ADEN network
### Annealing Schedule
Progressive sharpening of cluster assignments:

$β ← β × τ$  (multiplicative growth)
$Y ← Y + η·∇F_β(Y)$  (gradient descent on free energy)


## 📁 Project Structure

```
├── ADEN.py                    # Adaptive Distance Estimation Network
├── ADENTrain.py              # Training algorithms (TrainDbar, TrainY, TrainAnneal)
├── Env.py                    # Clustering environments (NumPy/PyTorch)
├── ClusteringGroundTruth.py  # Analytical ground truth solvers
├── TestCaseGenerator.py     # Synthetic dataset generation
├── benchmark.py             # Comprehensive benchmarking suite
├── benchmark_UDT.py         # Focused benchmark scenarios
├── utils.py                 # Utility functions (distances, seeding)
├── Plotter.py              # Visualization utilities
├── animator.py             # GIF animation generation
├── ReinforcementClustering.py # Classical tabular RL approach
├── DeepClusteringParametrized.ipynb # Main experiment notebook
├── TabularRL_Clustering.ipynb      # Classical RL experiments
├── Clustering_GT.ipynb            # Ground truth analysis
├── Benchmark/                     # Benchmark results (timestamped)
├── BenchmarkUDT/                 # UDT-specific results
├── Results/                      # Visualization outputs
└── animations/                   # Generated GIF animations
```

## 🎨 Visualization

The system provides comprehensive visualization capabilities:

- **Static Plots**: `Plotter.py` generates publication-ready clustering visualizations
- **Animations**: `animator.py` creates GIF animations showing clustering evolution
- **Real-time Monitoring**: Training progress with loss curves and convergence metrics

Example visualization code:
```python
from Plotter import PlotClustering

PlotClustering(
    X.cpu().numpy(), Y_opt.cpu().numpy(), pi_opt,
    figsize=(12, 6), cmap="gist_rainbow",
    save_path="Results/clustering_result.png"
)
```

## 🔬 Research Applications

This framework has been applied to:
- **Sensor Network Optimization**: UTD19 London sensor placement dataset

- **Synthetic Benchmark Problems**: Multi-modal, multi-scale clustering scenarios
- **Decentralized Systems**: Autonomous agent coordination and resource allocation

## 📈 Performance Metrics

The system uses multiple clustering quality metrics:
- **Chamfer Distance**: Bidirectional point-to-cluster matching
- **Hungarian Distance**: Optimal cluster center assignment cost
- **Free Energy**: Thermodynamic clustering objective
- **Distortion**: Weighted sum of within-cluster distances

## 🛠️ Development Guidelines

### Adding New Environments
1. Extend `ClusteringEnvNumpy` or `ClusteringEnvTorch`
2. Implement `return_probabilities()` and `step()` methods
3. Update benchmark configurations

### Model Architecture Changes
1. Modify `ADEN` class in `ADEN.py`
2. Ensure compatibility with `TrainDbar` batching
3. Update `reset_weights()` for proper initialization
4. Test with different `d_model` configurations

### Custom Datasets
1. Add data loading function in `TestCaseGenerator.py`
2. Follow the format: `return X, M, T_P, N, d`
3. Normalize data to [0,1] range for stability

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{autonomy_aware_clustering_2024,
  title={Autonomy-Aware Clustering},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Troubleshooting

### Common Issues

**CUDA Memory Errors**: Reduce `batch_size_dbar` or `num_samples_in_batch_dbar`

**Convergence Issues**: 
- Adjust `beta_growth_rate` (try 1.05-1.2)
- Increase `perturbation_std` to escape local minima
- Check environment parameter ranges

**Training Instability**: 
- Use `%env CUDA_LAUNCH_BLOCKING=1` in notebooks for debugging
- Ensure `utils.set_seed(0)` is called before training
- Monitor loss curves for numerical issues

**Performance**: 
- Use PyTorch environments for GPU training
- NumPy environments for ground truth computation only
- Profile with `torch.profiler` for bottleneck identification

For more detailed troubleshooting, see the GitHub Issues page.