# models/attn.py

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def build_spatiotemporal_causal_mask(
    seq_len: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    构造时空展开后的因果 mask
    输入序列长度 seq_len = T * N
    其中 T 为时间步数，N 为节点数

    返回:
        mask: [seq_len, seq_len]，bool
              True 表示允许注意，False 表示屏蔽
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")
    if seq_len % num_nodes != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by num_nodes ({num_nodes})"
        )

    time_steps = seq_len // num_nodes
    time_mask = torch.tril(
        torch.ones(time_steps, time_steps, dtype=torch.bool, device=device)
    )  # [T, T]

    # 扩展为 [T, N, T, N] -> [T*N, T*N]
    mask = (
        time_mask.unsqueeze(1)
        .unsqueeze(3)
        .expand(time_steps, num_nodes, time_steps, num_nodes)
        .reshape(seq_len, seq_len)
    )
    return mask


class STSFullSelfAttention(nn.Module):
    """
    Full Self-Attention with spatiotemporal causal mask support
    输入:
        query/key/value: [L, B, C]
    输出:
        out: [B, L, C]
    """

    def __init__(
        self,
        d_model: int = 8,
        n_heads: int = 1,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
        self_attention: bool = False,
        num_nodes: Optional[int] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.kdim = kdim if kdim is not None else d_model
        self.vdim = vdim if vdim is not None else d_model
        self.qkv_same_dim = (self.kdim == d_model and self.vdim == d_model)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.scaling = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.self_attention = self_attention
        self.num_nodes = num_nodes

        assert self.self_attention, "Only support self attention"
        assert (not self.self_attention) or self.qkv_same_dim, (
            "Self-attention requires query, key and value to be the same size"
        )

        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(self.kdim, d_model, bias=bias)
        self.W_V = nn.Linear(self.vdim, d_model, bias=bias)
        self.fc_out = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.W_Q.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.W_K.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.W_V.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.W_Q.weight)
            nn.init.xavier_uniform_(self.W_K.weight)
            nn.init.xavier_uniform_(self.W_V.weight)

        nn.init.xavier_uniform_(self.fc_out.weight)

        if self.W_Q.bias is not None:
            nn.init.constant_(self.W_Q.bias, 0.0)
            nn.init.constant_(self.W_K.bias, 0.0)
            nn.init.constant_(self.W_V.bias, 0.0)
            nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参数:
            query/key/value: [L, B, C]
            mask: [L, L] bool, True表示可见

        返回:
            out: [B, L, C]
        """
        L, B, C = query.shape
        assert C == self.d_model, f"query dim {C} != {self.d_model}"

        q = self.W_Q(query)
        k = self.W_K(key)
        v = self.W_V(value)

        q = q * self.scaling

        q = q.contiguous().view(L, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)  # [B,H,L,D]
        k = k.contiguous().view(L, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.contiguous().view(L, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)

        scores = torch.matmul(q, k.transpose(-2, -1))  # [B,H,L,L]

        if mask is None:
            if self.num_nodes is None:
                raise ValueError("num_nodes must be provided when mask is None.")
            mask = build_spatiotemporal_causal_mask(
                seq_len=L,
                num_nodes=self.num_nodes,
                device=query.device,
            )

        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # [B,H,L,D]
        context = context.permute(2, 0, 1, 3).contiguous().view(L, B, C)
        out = self.fc_out(context).transpose(0, 1)  # [B,L,C]
        return out


class STSProbSparseSelfAttention(nn.Module):
    """
    ProbSparse Self-Attention
    输入:
        query/key/value: [L, B, C]
    输出:
        out: [B, L, C]
    """

    def __init__(
        self,
        d_model: int = 8,
        n_heads: int = 1,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        factor: int = 5,
        dropout: float = 0.1,
        bias: bool = True,
        output_attention: bool = False,
        self_attention: bool = False,
        num_nodes: Optional[int] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.kdim = kdim if kdim is not None else d_model
        self.vdim = vdim if vdim is not None else d_model
        self.qkv_same_dim = (self.kdim == d_model and self.vdim == d_model)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.scaling = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention
        self.self_attention = self_attention
        self.factor = factor
        self.num_nodes = num_nodes

        assert self.self_attention, "Only support self attention"
        assert (not self.self_attention) or self.qkv_same_dim, (
            "Self-attention requires Q,K,V same size"
        )

        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(self.kdim, d_model, bias=bias)
        self.W_V = nn.Linear(self.vdim, d_model, bias=bias)
        self.fc_out = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

        if self.W_Q.bias is not None:
            nn.init.constant_(self.W_Q.bias, 0.0)
            nn.init.constant_(self.W_K.bias, 0.0)
            nn.init.constant_(self.W_V.bias, 0.0)
            nn.init.constant_(self.fc_out.bias, 0.0)

    def _prob_QK(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        sample_k: int,
        n_top: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Q, K: [B,H,L,D]
        返回:
            scores_top: [B,H,n_top,L]
            top_indices: [B,H,n_top]
        """
        B, H, L, D = Q.shape
        device = Q.device

        index_sample = torch.randint(L, (L, sample_k), device=device)  # [L, sample_k]

        K_expand = K.unsqueeze(2).expand(B, H, L, L, D)  # [B,H,L,L,D]
        K_sample = K_expand.gather(
            3,
            index_sample.view(1, 1, L, sample_k, 1).expand(B, H, L, sample_k, D),
        )  # [B,H,L,sample_k,D]

        Q_expand = Q.unsqueeze(3).expand(B, H, L, sample_k, D)  # [B,H,L,sample_k,D]
        Q_K_sample = (Q_expand * K_sample).sum(dim=-1)  # [B,H,L,sample_k]

        sparsity_measure = Q_K_sample.max(dim=-1).values - Q_K_sample.mean(dim=-1)  # [B,H,L]
        top_indices = sparsity_measure.topk(n_top, dim=-1).indices  # [B,H,n_top]

        Q_reduce = Q.gather(2, top_indices.unsqueeze(-1).expand(B, H, n_top, D))  # [B,H,n_top,D]
        scores_top = torch.matmul(Q_reduce, K.transpose(-2, -1))  # [B,H,n_top,L]

        return scores_top, top_indices

    def _get_initial_context(self, V: torch.Tensor) -> torch.Tensor:
        """
        V: [B,H,L,D]
        返回:
            context: [B,H,L,D]
        """
        context = V.mean(dim=2, keepdim=True).expand_as(V).clone()
        return context

    def _update_context(
        self,
        context: torch.Tensor,
        V: torch.Tensor,
        scores_top: torch.Tensor,
        top_indices: torch.Tensor,
        full_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        context: [B,H,L,D]
        V: [B,H,L,D]
        scores_top: [B,H,n_top,L]
        top_indices: [B,H,n_top]
        full_mask: [L,L] bool
        """
        B, H, L, D = V.shape
        n_top = scores_top.size(2)

        if full_mask is not None:
            # 为每个 top query 取出对应的一行 mask
            selected_mask = full_mask[top_indices.reshape(-1)]  # [(B*H*n_top), L]
            selected_mask = selected_mask.view(B, H, n_top, L)
            scores_top = scores_top.masked_fill(~selected_mask, float("-inf"))

        attn = torch.softmax(scores_top, dim=-1)
        attn = self.dropout(attn)  # [B,H,n_top,L]

        updates = torch.matmul(attn, V)  # [B,H,n_top,D]
        context.scatter_(
            2,
            top_indices.unsqueeze(-1).expand(B, H, n_top, D),
            updates,
        )

        if self.output_attention:
            dense_attn = torch.zeros(B, H, L, L, device=V.device, dtype=attn.dtype)
            dense_attn.scatter_(
                2,
                top_indices.unsqueeze(-1).expand(B, H, n_top, L),
                attn,
            )
            return context, dense_attn

        return context, None

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query/key/value: [L,B,C]
        mask: [L,L] bool
        返回:
            out: [B,L,C]
        """
        L, B, C = query.shape
        assert C == self.d_model, f"query dim {C} != {self.d_model}"

        q = self.W_Q(query).transpose(0, 1)  # [B,L,C]
        k = self.W_K(key).transpose(0, 1)
        v = self.W_V(value).transpose(0, 1)

        q = q.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,H,L,D]
        k = k.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        q = q * self.scaling

        if mask is None:
            if self.num_nodes is None:
                raise ValueError("num_nodes must be provided when mask is None.")
            mask = build_spatiotemporal_causal_mask(
                seq_len=L,
                num_nodes=self.num_nodes,
                device=query.device,
            )

        sample_k = min(self.factor * int(math.ceil(math.log(L + 1))), L)
        n_top = min(self.factor * int(math.ceil(math.log(L + 1))), L)

        scores_top, top_indices = self._prob_QK(q, k, sample_k=sample_k, n_top=n_top)
        context = self._get_initial_context(v)
        context, _ = self._update_context(
            context=context,
            V=v,
            scores_top=scores_top,
            top_indices=top_indices,
            full_mask=mask,
        )

        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        out = self.fc_out(context)  # [B,L,C]
        return out