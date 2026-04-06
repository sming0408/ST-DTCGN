from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attn import (
    STSFullSelfAttention,
    STSProbSparseSelfAttention,
    build_spatiotemporal_causal_mask,
)


# =========================================================
# Normalization Modules
# =========================================================
class DyTanhNorm(nn.Module):
    """
    DyTanhNorm(x) = gamma * tanh(alpha * x) + beta
    """
    def __init__(self, normalized_shape, channels_last=True, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last

        if isinstance(normalized_shape, int):
            shape = normalized_shape
        else:
            shape = normalized_shape[0]

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        #x = torch.tanh(self.alpha * x)
        #x = F.gelu(self.alpha * x)
        #x = F.relu(self.alpha * x)
        x = x * torch.sigmoid(self.alpha * x)
        if self.channels_last:
            return x * self.weight + self.bias
        return x * self.weight[:, None, None] + self.bias[:, None, None]

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"channels_last={self.channels_last}"
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def build_norm(norm_type: str, dim: int):
    norm_type = norm_type.strip()

    if norm_type == "LayerNorm":
        return nn.LayerNorm(dim)
    elif norm_type == "RMSNorm":
        return RMSNorm(dim)
    elif norm_type == "DyTanhNorm":
        return DyTanhNorm(dim, channels_last=True)
    else:
        raise ValueError(
            f"Unsupported norm_type: {norm_type}. "
            f"Choose from ['LayerNorm', 'RMSNorm', 'DyTanhNorm']"
        )


# =========================================================
# Feed-Forward Activations for direct comparison
# =========================================================
class FeedForward(nn.Module):
    """
    Supports:
    - relu
    - gelu
    - tanh
    - swiglu
    """
    def __init__(
        self,
        d_model: int,
        forward_expansion: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.activation = activation.lower()
        hidden_dim = forward_expansion * d_model

        if self.activation in {"relu", "gelu", "tanh"}:
            self.fc1 = nn.Linear(d_model, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, d_model)
        elif self.activation == "swiglu":
            self.fc1 = nn.Linear(d_model, hidden_dim * 2)
            self.fc2 = nn.Linear(hidden_dim, d_model)
        else:
            raise ValueError(
                f"Unsupported ffn_activation: {activation}. "
                f"Choose from ['relu', 'gelu', 'tanh', 'swiglu']"
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

        if self.activation == "gelu":
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

        if self.activation == "tanh":
            x = self.fc1(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

        if self.activation == "swiglu":
            x = self.fc1(x)
            x1, x2 = torch.chunk(x, 2, dim=-1)
            x = F.silu(x1) * x2
            x = self.dropout(x)
            x = self.fc2(x)
            return x

        raise ValueError(f"Unsupported activation: {self.activation}")




# =========================================================
# Utility for theoretical / empirical proxy statistics
# =========================================================
def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        return {
            "mean_abs": x.abs().mean().item(),
            "std": x.std(unbiased=False).item(),
            "max_abs": x.abs().max().item(),
            "l2_norm": torch.norm(x, p=2).item() / max(x.numel(), 1) ** 0.5,
        }


def drift_ratio(x_prev: torch.Tensor, x_next: torch.Tensor, eps: float = 1e-8) -> float:
    with torch.no_grad():
        num = torch.norm(x_next - x_prev, p=2)
        den = torch.norm(x_prev, p=2) + eps
        return (num / den).item()


# =========================================================
# Dytrans block with configurable norm + activation
# =========================================================
class Dytrans(nn.Module):
    def __init__(
        self,
        d_model: int = 8,
        n_heads: int = 1,
        factor: int = 10,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
        dropout: float = 0.1,
        forward_expansion: int = 64,
        use_informer: bool = True,
        num_nodes: Optional[int] = None,
        norm_type: str = "DyTanhNorm",
        ffn_activation: str = "relu",
        return_stability_stats: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention
        self.use_informer = use_informer
        self.num_nodes = num_nodes
        self.norm_type = norm_type
        self.ffn_activation = ffn_activation.lower()
        self.return_stability_stats = return_stability_stats

        if use_informer:
            self.self_attention = STSProbSparseSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                factor=factor,
                dropout=attention_dropout,
                output_attention=output_attention,
                self_attention=True,
                num_nodes=num_nodes,
            )
        else:
            self.self_attention = STSFullSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=attention_dropout,
                self_attention=True,
                num_nodes=num_nodes,
            )

        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)

        self.ffn = FeedForward(
            d_model=d_model,
            forward_expansion=forward_expansion,
            dropout=dropout,
            activation=ffn_activation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        query: [T*N, B, C]
        mask: [T*N, T*N] bool
        """
        stats = None

        norm_q = self.norm1(query)

        self_attention = self.self_attention(
            query=norm_q,
            key=norm_q,
            value=norm_q,
            mask=mask,
        )

        x_residual_base = query.permute(1, 0, 2)
        x = self.dropout(self_attention) + x_residual_base

        norm_x = self.norm2(x)
        ff = self.ffn(norm_x)
        out = self.dropout(ff) + x

        if self.return_stability_stats:
            stats = {
                "input": tensor_stats(query),
                "after_norm1": tensor_stats(norm_q),
                "after_attn_residual": tensor_stats(x),
                "after_norm2": tensor_stats(norm_x),
                "after_ffn": tensor_stats(ff),
                "output": tensor_stats(out),
                "drift_input_to_attnres": drift_ratio(x_residual_base, x),
                "drift_attnres_to_output": drift_ratio(x, out),
            }

        out = out.permute(1, 0, 2)

        if self.return_stability_stats:
            return out, stats
        return out


# =========================================================
# Adaptive Graph
# =========================================================
class AdaptiveUnifiedGraph(nn.Module):
    """
    A_unified = alpha * A_static + (1 - alpha) * A_adaptive
    """

    def __init__(self, static_adj: torch.Tensor, num_nodes: int, embed_dim: int = 16):
        super().__init__()
        self.num_nodes = num_nodes

        static_adj = static_adj.float()
        self.register_buffer("A_static", static_adj)

        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.node_emb2 = nn.Parameter(torch.randn(embed_dim, num_nodes))
        self.alpha = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def normalize_adj(A: torch.Tensor) -> torch.Tensor:
        A = F.relu(A)
        A = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        degree = A.sum(dim=-1)
        d_inv_sqrt = torch.pow(degree.clamp(min=1e-12), -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self) -> torch.Tensor:
        A_adaptive = F.softmax(F.relu(self.node_emb1 @ self.node_emb2), dim=-1)
        alpha = torch.sigmoid(self.alpha)

        A_static_norm = self.normalize_adj(self.A_static)
        A_adaptive_norm = self.normalize_adj(A_adaptive)

        A_unified = alpha * A_static_norm + (1.0 - alpha) * A_adaptive_norm
        return A_unified


# =========================================================
# GCN Operation
# =========================================================
class gcn_operation(nn.Module):
    def __init__(
        self,
        adj: torch.Tensor,
        in_dim: int,
        out_dim: int,
        num_vertices: int,
        activation: str = "GLU",
        use_adaptive_graph: bool = True,
        adaptive_embed_dim: int = 16,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation
        assert activation in {"GLU", "relu"}

        self.use_adaptive_graph = use_adaptive_graph
        if use_adaptive_graph:
            self.graph_builder = AdaptiveUnifiedGraph(
                static_adj=adj,
                num_nodes=num_vertices,
                embed_dim=adaptive_embed_dim,
            )
        else:
            self.register_buffer("static_adj", adj.float())

        if activation == "GLU":
            self.FC = nn.Linear(in_dim, 2 * out_dim, bias=True)
        else:
            self.FC = nn.Linear(in_dim, out_dim, bias=True)

    @staticmethod
    def block_diag_adj(adj: torch.Tensor, T: int) -> torch.Tensor:
        return torch.block_diag(*[adj for _ in range(T)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: [T*N, B, Cin]
        mask: [T*N, T*N] bool
        """
        seq_len, batch_size, _ = x.shape
        if seq_len % self.num_vertices != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by num_vertices ({self.num_vertices})"
            )
        T = seq_len // self.num_vertices

        if self.use_adaptive_graph:
            base_adj = self.graph_builder()
        else:
            base_adj = self.static_adj.to(x.device)

        adj_block = self.block_diag_adj(base_adj.to(x.device), T=T)

        if mask is not None:
            adj_block = adj_block * mask.float()

        x = torch.einsum("nm,mbc->nbc", adj_block, x)

        if self.activation == "GLU":
            lhs_rhs = self.FC(x)
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)
            return lhs * torch.sigmoid(rhs)

        return torch.relu(self.FC(x))


# =========================================================
# ST_DTGCNM
# =========================================================
class ST_DTGCNM(nn.Module):
    def __init__(
        self,
        adj: torch.Tensor,
        in_dim: int,
        out_dims: List[int],
        num_of_vertices: int,
        d_model: int,
        n_heads: int,
        factor: int,
        attention_dropout: float,
        output_attention: bool,
        dropout: float,
        forward_expansion: int,
        activation: str = "GLU",
        use_transformer: bool = True,
        use_informer: bool = True,
        use_adaptive_graph: bool = True,
        norm_type: str = "DyTanhNorm",
        ffn_activation: str = "relu",
        return_stability_stats: bool = False,
    ):
        super().__init__()

        self.adj = adj
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.use_transformer = use_transformer
        self.use_informer = use_informer
        self.norm_type = norm_type
        self.ffn_activation = ffn_activation
        self.return_stability_stats = return_stability_stats

        self.gcn_operations = nn.ModuleList()
        self.ST_GCN = nn.ModuleList()

        current_dim = in_dim
        for out_dim in out_dims:
            self.gcn_operations.append(
                gcn_operation(
                    adj=self.adj,
                    in_dim=current_dim,
                    out_dim=out_dim,
                    num_vertices=self.num_of_vertices,
                    activation=self.activation,
                    use_adaptive_graph=use_adaptive_graph,
                )
            )

            self.ST_GCN.append(
                Dytrans(
                    d_model=current_dim,
                    n_heads=n_heads,
                    factor=factor,
                    attention_dropout=attention_dropout,
                    output_attention=output_attention,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    use_informer=use_informer,
                    num_nodes=num_of_vertices,
                    norm_type=norm_type,
                    ffn_activation=ffn_activation,
                    return_stability_stats=return_stability_stats,
                )
            )
            current_dim = out_dim

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: [T*N, B, Cin]
        """
        input_transformer = x
        block_stats = []

        if not self.use_transformer:
            raise NotImplementedError("Current implementation requires use_transformer=True.")

        for i in range(len(self.out_dims)):
            if self.return_stability_stats:
                transformer_temp, stats = self.ST_GCN[i](input_transformer, mask)
                block_stats.append(stats)
            else:
                transformer_temp = self.ST_GCN[i](input_transformer, mask)

            transformer_temp = self.gcn_operations[i](transformer_temp, mask)

            if transformer_temp.shape[-1] == input_transformer.shape[-1]:
                input_transformer = transformer_temp + input_transformer
            else:
                input_transformer = transformer_temp

        if self.return_stability_stats:
            return input_transformer, block_stats
        return input_transformer


# =========================================================
# Temporal Branch
# =========================================================
class TemporalInception(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.branch1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            padding=(0, 0),
        )
        self.branch2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 3),
            dilation=(1, 1),
            padding=(0, 1),
        )
        self.branch3 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 3),
            dilation=(1, 2),
            padding=(0, 2),
        )
        self.branch4 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 3),
            dilation=(1, 4),
            padding=(0, 4),
        )

        self.proj = nn.Conv2d(
            in_channels=channels * 4,
            out_channels=channels,
            kernel_size=(1, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, N, T]
        """
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.proj(out)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out


class GatedFusion(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(channels * 2, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_main: torch.Tensor, x_branch: torch.Tensor) -> torch.Tensor:
        """
        x_main:   [B, T, N, C]
        x_branch: [B, T, N, C]
        """
        fused = torch.cat([x_main, x_branch], dim=-1)
        gate = torch.sigmoid(self.gate_proj(fused))
        out = gate * x_main + (1.0 - gate) * x_branch
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


# =========================================================
# ST_DTGCNL
# =========================================================
class ST_DTGCNL(nn.Module):
    def __init__(
        self,
        adj: torch.Tensor,
        history: int,
        num_of_vertices: int,
        in_dim: int,
        out_dims: List[int],
        d_model: int,
        n_heads: int,
        factor: int,
        attention_dropout: float,
        output_attention: bool,
        dropout: float,
        forward_expansion: int,
        strides: int = 12,
        activation: str = "GLU",
        temporal_emb: bool = True,
        spatial_emb: bool = True,
        use_transformer: bool = True,
        use_informer: bool = False,
        use_adaptive_graph: bool = True,
        norm_type: str = "DyTanhNorm",
        ffn_activation: str = "relu",
        return_stability_stats: bool = False,
        use_temporal_inception=True,
    ):
        super().__init__()

        self.adj = adj
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention
        self.dropout = dropout
        self.forward_expansion = forward_expansion
        self.use_transformer = use_transformer
        self.use_informer = use_informer
        self.norm_type = norm_type
        self.ffn_activation = ffn_activation
        self.return_stability_stats = return_stability_stats
        self.use_temporal_inception = use_temporal_inception

        self.input_proj = nn.Linear(in_dim, d_model)

        self.ST_ITGCNMS = ST_DTGCNM(
            adj=adj,
            in_dim=d_model,
            out_dims=out_dims,
            num_of_vertices=num_of_vertices,
            d_model=d_model,
            n_heads=n_heads,
            factor=factor,
            attention_dropout=attention_dropout,
            output_attention=output_attention,
            dropout=dropout,
            forward_expansion=forward_expansion,
            activation=activation,
            use_transformer=use_transformer,
            use_informer=use_informer,
            use_adaptive_graph=use_adaptive_graph,
            norm_type=norm_type,
            ffn_activation=ffn_activation,
            return_stability_stats=return_stability_stats,
        )

        if temporal_emb:
            self.base_temporal_emb = nn.Parameter(torch.FloatTensor(1, history, 1, d_model))
            self.weekly_emb = nn.Parameter(torch.FloatTensor(1, 7, 1, d_model))
            self.monthly_emb = nn.Parameter(torch.FloatTensor(1, 30, 1, d_model))

        if spatial_emb:
            self.base_spatial_emb = nn.Parameter(torch.FloatTensor(1, 1, num_of_vertices, d_model))
            self.risk_spatial_emb = nn.Parameter(torch.FloatTensor(1, 1, num_of_vertices, d_model))
            self.alpha = nn.Parameter(torch.FloatTensor(1))
        
        if self.use_temporal_inception:
            self.parallel_temporal = TemporalInception(
                channels=d_model,
                dropout=dropout,
            )
            self.branch_proj = nn.Linear(d_model, out_dims[-1])
            self.gated_fusion = GatedFusion(
                channels=out_dims[-1],
                dropout=dropout,
            )
            
        self._reset_parameters()

    def _reset_parameters(self):
        if self.temporal_emb:
            nn.init.xavier_uniform_(self.base_temporal_emb)
            nn.init.xavier_uniform_(self.weekly_emb)
            nn.init.xavier_uniform_(self.monthly_emb)

        if self.spatial_emb:
            nn.init.xavier_uniform_(self.base_spatial_emb)
            nn.init.xavier_uniform_(self.risk_spatial_emb)
            nn.init.constant_(self.alpha, 0.5)

    def _add_temporal_embedding(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        x = x + self.base_temporal_emb[:, :T]

        weekly_idx = torch.arange(T, device=x.device) % 7
        x = x + self.weekly_emb[:, weekly_idx]

        monthly_idx = torch.arange(T, device=x.device) % 30
        x = x + self.monthly_emb[:, monthly_idx]

        return x

    def _add_spatial_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.base_spatial_emb
        x = x + torch.sigmoid(self.alpha) * self.risk_spatial_emb
        return x

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, N, Cin]
        return:
            out: [B, T, N, Cout]
        """
        B, T, N, Cin = x.shape
        assert N == self.num_of_vertices, (
            f"num_of_vertices mismatch: input {N}, expected {self.num_of_vertices}"
        )

        x = self.input_proj(x)

        if self.temporal_emb:
            x = self._add_temporal_embedding(x)

        if self.spatial_emb:
            x = self._add_spatial_embedding(x)

        seq_len = T * N
        mask = build_spatiotemporal_causal_mask(
            seq_len=seq_len,
            num_nodes=N,
            device=x.device,
        )

        x_main = x.reshape(B, T * N, self.d_model).permute(1, 0, 2)

        layer_stats = None
        if self.return_stability_stats:
            x_main, layer_stats = self.ST_ITGCNMS(x_main, mask=mask)
        else:
            x_main = self.ST_ITGCNMS(x_main, mask=mask)

        x_main = x_main.permute(1, 0, 2).reshape(B, T, N, self.out_dims[-1])
        
        if self.use_temporal_inception:
            x_branch = x.permute(0, 3, 2, 1)  # [B,C,N,T]
            x_branch = self.parallel_temporal(x_branch)
            
            if x_branch.size(-1) != T:
                x_branch = F.interpolate(
                    x_branch,
                    size=(N, T),
                    mode="bilinear",
                    align_corners=False,
                )
            x_branch = x_branch.permute(0, 3, 2, 1)
            x_branch = self.branch_proj(x_branch)
            out = self.gated_fusion(x_main, x_branch)
        else:
            #🚨 消融：只用主干
            out = x_main

        if self.return_stability_stats:
            return out, layer_stats
        return out


# =========================================================
# Output Layer
# =========================================================
class output_layer(nn.Module):
    def __init__(
        self,
        history: int,
        num_of_vertices: int,
        in_dim: int,
        hidden_dim: int,
        horizon: int,
        dropout: float,
    ):
        super().__init__()
        self.history = history
        self.num_of_vertices = num_of_vertices
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.fc1 = nn.Linear(history * in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(num_of_vertices)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_of_vertices)

        self.fc3 = nn.Linear(hidden_dim, horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, N, C]
        return: [B, horizon, N]
        """
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, N, T * C)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = x.permute(0, 2, 1)
        return x


# =========================================================
# Main Model
# =========================================================
class ST_DTGCN(nn.Module):
    def __init__(
        self,
        adj: torch.Tensor,
        history: int,
        num_of_vertices: int,
        in_dim: int,
        hidden_dims: List[List[int]],
        first_layer_embedding_size: int,
        out_layer_dim: int,
        d_model: int,
        n_heads: int,
        factor: int,
        attention_dropout: float,
        output_attention: bool,
        dropout: float,
        forward_expansion: int,
        horizon: int,
        strides: int = 12,
        activation: str = "GLU",
        temporal_emb: bool = True,
        spatial_emb: bool = True,
        use_transformer: bool = True,
        use_informer: bool = False,
        use_adaptive_graph: bool = True,
        norm_type: str = "DyTanhNorm",
        ffn_activation: str = "relu",
        return_stability_stats: bool = False,
        use_temporal_inception: bool = True
    ):
        super().__init__()

        self.history = history
        self.num_of_vertices = num_of_vertices
        self.in_dim = in_dim
        self.horizon = horizon
        self.norm_type = norm_type
        self.ffn_activation = ffn_activation
        self.return_stability_stats = return_stability_stats

        self.input_conv_1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=first_layer_embedding_size,
            kernel_size=(1, 1),
        )

        self.layers = nn.ModuleList()
        current_in_dim = first_layer_embedding_size

        for out_dims in hidden_dims:
            layer = ST_DTGCNL(
                adj=adj,
                history=history,
                num_of_vertices=num_of_vertices,
                in_dim=current_in_dim,
                out_dims=out_dims,
                d_model=d_model,
                n_heads=n_heads,
                factor=factor,
                attention_dropout=attention_dropout,
                output_attention=output_attention,
                dropout=dropout,
                forward_expansion=forward_expansion,
                strides=strides,
                activation=activation,
                temporal_emb=temporal_emb,
                spatial_emb=spatial_emb,
                use_transformer=use_transformer,
                use_informer=use_informer,
                use_adaptive_graph=use_adaptive_graph,
                norm_type=norm_type,
                ffn_activation=ffn_activation,
                return_stability_stats=return_stability_stats,
                use_temporal_inception=use_temporal_inception,
            )
            self.layers.append(layer)
            current_in_dim = out_dims[-1]

        self.predictor = output_layer(
            history=history,
            num_of_vertices=num_of_vertices,
            in_dim=current_in_dim,
            hidden_dim=out_layer_dim,
            horizon=horizon,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, N, Cin]
        return:
            out: [B, H, N]
            or
            (out, stats)
        """
        x = x.permute(0, 3, 2, 1)
        x = self.input_conv_1(x)
        x = x.permute(0, 3, 2, 1)

        all_stats = []
        for layer in self.layers:
            if self.return_stability_stats:
                x, stats = layer(x)
                all_stats.append(stats)
            else:
                x = layer(x)

        out = self.predictor(x)

        if self.return_stability_stats:
            return out, all_stats
        return out