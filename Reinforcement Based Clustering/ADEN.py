import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MultiHeadDistanceAttention(nn.Module):
    """Multi-head attention mechanism for computing adaptive distance relationships"""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(float(self.d_k))

    def _transform(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        # queries: (batch_size, n_queries, d_model)
        # keys: (batch_size, n_keys, d_model)
        # values: (batch_size, n_keys, d_model)

        batch_size = queries.size(0)

        # Linear transformations and reshape
        Q = self._transform(
            self.W_q(queries), batch_size
        )  # (batch_size, n_heads, n_queries, d_k)
        K = self._transform(
            self.W_k(keys), batch_size
        )  # (batch_size, n_heads, n_keys, d_k)
        V = self._transform(
            self.W_v(values), batch_size
        )  # (batch_size, n_heads, n_keys, d_k)

        # Scaled dot-product attention
        scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        )  # (batch_size, n_heads, n_queries, n_keys)
        attn_weights = F.softmax(
            scores, dim=-1
        )  # (batch_size, n_heads, n_queries, n_keys)
        attn_weights = self.dropout(
            attn_weights
        )  # (batch_size, n_heads, n_queries, n_keys)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (batch_size, n_heads, n_queries, d_k)

        # Concatenate heads and put through final linear layer
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )  # (batch_size, n_queries, d_model)

        return self.W_o(context)  # (batch_size, n_queries, d_model)


class AdaptiveDistanceBlock(nn.Module):
    """Core block that learns adaptive distance deviations"""

    def __init__(self, d_model: int, d_ff: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadDistanceAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network with GELU activation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, data_emb: torch.Tensor, cluster_emb: torch.Tensor
    ) -> torch.Tensor:
        # Self-attention on data points with cluster centers as context
        attn_out = self.attention(data_emb, cluster_emb, cluster_emb)
        data_emb = self.norm1(data_emb + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(data_emb)
        data_emb = self.norm2(data_emb + ffn_out)

        return data_emb



class ADEN(nn.Module):
    """
    Adaptive Distance Estimation Network

    A sophisticated architecture for learning adaptive distances in clustering tasks.
    Combines transformer-style attention, learnable metrics, and residual connections.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_metric_tensor: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.use_metric_tensor = use_metric_tensor

        # Input projections
        self.data_projection = nn.Linear(input_dim, d_model)
        self.cluster_projection = nn.Linear(input_dim, d_model)

        # Stack of adaptive distance blocks
        self.blocks = nn.ModuleList(
            [
                AdaptiveDistanceBlock(d_model, d_ff, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )


        # Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.distance_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # Temperature parameter for distance scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_base_distances(
        self, data_points: torch.Tensor, cluster_centers: torch.Tensor
    ) -> torch.Tensor:
        """Compute base squared Euclidean distances"""
        # data_points: (batch_size, N, d)
        # cluster_centers: (batch_size, M, d)

        data_expanded = data_points.unsqueeze(2)  # (batch_size, N, 1, d)
        centers_expanded = cluster_centers.unsqueeze(1)  # (batch_size, 1, M, d)

        # Squared Euclidean distance
        diff = data_expanded - centers_expanded  # (batch_size, N, M, d)
        base_distances = torch.sum(diff**2, dim=-1)  # (batch_size, N, M)

        return base_distances

    def forward(
        self, data_points: torch.Tensor, cluster_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            data_points: (batch_size, N, input_dim)
            cluster_centers: (batch_size, M, input_dim)

        Returns:
            distances: (batch_size, N, M) - Adaptive distance matrix
        """
        _, N, _ = data_points.shape
        _, M, _ = cluster_centers.shape

        # Project to model dimension
        data_emb = self.data_projection(data_points)
        cluster_emb = self.cluster_projection(cluster_centers)

        # Pass through adaptive distance blocks
        for block in self.blocks:
            data_emb = block(data_emb, cluster_emb)

        data_emb = self.output_norm(data_emb)

        # Compute pairwise features for distance prediction
        data_expanded = data_emb.unsqueeze(2).expand(-1, -1, M, -1)
        cluster_expanded = cluster_emb.unsqueeze(1).expand(-1, N, -1, -1)

        # Concatenate features
        pair_features = torch.cat([data_expanded, cluster_expanded], dim=-1)

        # Predict distance deviations
        distance_deviations = self.distance_head(pair_features).squeeze(-1)

        # Compute base distances
        base_distances = self.compute_base_distances(data_points, cluster_centers)

        # Final adaptive distances
        adaptive_distances = base_distances + self.temperature * distance_deviations

        # Ensure non-negative distances
        adaptive_distances = F.softplus(adaptive_distances)

        return adaptive_distances


class ADENLoss(nn.Module):
    """Custom loss function for ADEN training"""

    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha  # Weight for distance prediction loss
        self.beta = beta  # Weight for regularization

    def forward(
        self,
        predicted_distances: torch.Tensor,
        target_distances: torch.Tensor,
        model: ADEN,
    ) -> torch.Tensor:
        # Main distance prediction loss
        distance_loss = F.mse_loss(predicted_distances, target_distances)

        # Regularization on metric tensor
        reg_loss = 0.0
        if model.use_metric_tensor:
            U, V = model.metric_tensor.U, model.metric_tensor.V
            reg_loss = torch.norm(U, "fro") + torch.norm(V, "fro")

        # Temperature regularization
        temp_reg = torch.abs(model.temperature - 1.0)

        total_loss = self.alpha * distance_loss + self.beta * (reg_loss + temp_reg)

        return total_loss


def create_sample_data(
    batch_size: int = 32, N: int = 100, M: int = 10, d: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample data for testing"""
    data_points = torch.randn(batch_size, N, d)
    cluster_centers = torch.randn(batch_size, M, d)

    # Generate target distances (base + some learned pattern)
    base_dist = torch.cdist(data_points, cluster_centers, p=2) ** 2
    noise = 0.1 * torch.randn_like(base_dist)
    target_distances = base_dist + noise

    return data_points, cluster_centers, target_distances


def train_aden():
    """Example training loop"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = ADEN(
        input_dim=64, d_model=256, n_layers=4, n_heads=8, d_ff=1024, dropout=0.1
    ).to(device)

    # Loss and optimizer
    criterion = ADENLoss(alpha=1.0, beta=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    model.train()
    for epoch in range(100):
        # Generate batch data
        data_points, cluster_centers, target_distances = create_sample_data()
        data_points = data_points.to(device)
        cluster_centers = cluster_centers.to(device)
        target_distances = target_distances.to(device)

        # Forward pass
        predicted_distances = model(data_points, cluster_centers)
        loss = criterion(predicted_distances, target_distances, model)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


if __name__ == "__main__":
    # Test the model
    model = ADEN(input_dim=64, d_model=256, n_layers=4)
    data_points, cluster_centers, _ = create_sample_data(batch_size=2, N=50, M=5)

    with torch.no_grad():
        output = model(data_points, cluster_centers)
        print(f"Input data shape: {data_points.shape}")
        print(f"Cluster centers shape: {cluster_centers.shape}")
        print(f"Output distance matrix shape: {output.shape}")
        print(f"Sample distances:\n{output[0, :5, :].numpy()}")
