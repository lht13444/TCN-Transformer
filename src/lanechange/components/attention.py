"""Projected dense and sampled-query ProbSparse-style attention components."""

import math

import torch
from torch import Tensor, nn

from .masking import require_observations, valid_mask, zero_invalid


class ProjectedAttention(nn.Module):
    """Attention over observed features; True mask entries are valid.

    The sparse backend samples keys before ranking queries. Only selected
    queries form scores against all valid keys; other queries start with masked
    mean V. max-minus-sample-mean is the sparsity proxy in this independent
    approximation, rather than a verbatim implementation of Informer.

    Evaluation sampling is deterministic and independent of batch order.
    ``auto`` routes each sample by effective length; the default is
    explicitly ``probsparse``. ``last_mode`` reports the actual backend, with
    ``mixed`` indicating a batch containing both dense and sparse samples.
    """

    def __init__(
        self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1,
        mode: str = "probsparse", sparse_factor: float = 3.0,
        sparse_threshold: int = 64, eval_seed: int = 1729,
    ) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1 or d_model % n_heads:
            raise ValueError("attention dimensions must be positive and divisible")
        if mode not in {"dense", "probsparse", "auto"}:
            raise ValueError("mode must be dense, probsparse, or auto")
        if not math.isfinite(sparse_factor) or sparse_factor <= 0:
            raise ValueError("sparse_factor must be finite and positive")
        if sparse_threshold < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid sparse threshold or dropout")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.mode = mode
        self.sparse_factor = sparse_factor
        self.sparse_threshold = sparse_threshold
        self.eval_seed = eval_seed
        self.last_mode = "uninitialized"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _project(self, x: Tensor, projection: nn.Linear) -> Tensor:
        return projection(x).view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

    def _dense(self, q: Tensor, k: Tensor, v: Tensor, key_valid: Tensor) -> Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~key_valid[:, None, None, :], -torch.inf)
        return torch.matmul(self.dropout(scores.softmax(dim=-1)), v)

    def _probsparse(self, q: Tensor, k: Tensor, v: Tensor, query_valid: Tensor, key_valid: Tensor) -> Tensor:
        outputs = []
        for batch_index in range(q.shape[0]):
            query_indices = query_valid[batch_index].nonzero(as_tuple=True)[0]
            key_indices = key_valid[batch_index].nonzero(as_tuple=True)[0]
            n_queries, n_keys = query_indices.numel(), key_indices.numel()
            if n_queries == 0:
                outputs.append(q[batch_index] * 0.0)
                continue
            query = q[batch_index].index_select(1, query_indices)
            key = k[batch_index].index_select(1, key_indices)
            value = v[batch_index].index_select(1, key_indices)
            sample_count = min(n_keys, max(1, math.ceil(self.sparse_factor * math.log(n_keys + 1))))
            top_count = min(n_queries, max(1, math.ceil(self.sparse_factor * math.log(n_queries + 1))))
            generator = None
            if not self.training:
                generator = torch.Generator(device=q.device)
                generator.manual_seed(self.eval_seed + 1009 * n_queries + 9176 * n_keys)
            sampled_indices = torch.randint(
                n_keys, (self.n_heads, n_queries, sample_count), device=q.device, generator=generator,
            )
            head_indices = torch.arange(self.n_heads, device=q.device)[:, None, None]
            sampled_key = key[head_indices, sampled_indices]
            sampled_scores = (query.unsqueeze(2) * sampled_key).sum(-1) / math.sqrt(self.head_dim)
            sparsity = sampled_scores.max(dim=-1).values - sampled_scores.mean(dim=-1)
            selected_indices = sparsity.topk(top_count, dim=-1, sorted=False).indices
            selected_query = query.gather(1, selected_indices.unsqueeze(-1).expand(-1, -1, self.head_dim))
            scores = torch.matmul(selected_query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            selected_context = torch.matmul(self.dropout(scores.softmax(dim=-1)), value)
            context = value.mean(dim=1, keepdim=True).expand(-1, n_queries, -1)
            context = context.scatter(1, selected_indices.unsqueeze(-1).expand(-1, -1, self.head_dim), selected_context)
            full_context = q[batch_index].new_zeros(self.n_heads, q.shape[2], self.head_dim)
            outputs.append(full_context.index_copy(1, query_indices, context))
        return torch.stack(outputs, dim=0)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor,
        query_valid: Tensor | None = None, key_valid: Tensor | None = None,
    ) -> Tensor:
        q_valid = valid_mask(query, query_valid, "attention query")
        k_valid = valid_mask(key, key_valid, "attention key")
        if key.shape != value.shape or query.shape[0] != key.shape[0]:
            raise ValueError("keys/values must match and batch sizes must agree")
        if query.shape[-1] != self.d_model or key.shape[-1] != self.d_model:
            raise ValueError("attention feature dimension must equal d_model")
        require_observations(k_valid)
        query, key, value = zero_invalid(query, q_valid), zero_invalid(key, k_valid), zero_invalid(value, k_valid)
        q, k, v = self._project(query, self.q_proj), self._project(key, self.k_proj), self._project(value, self.v_proj)
        mode = self.mode
        if mode == "auto":
            use_dense = torch.maximum(q_valid.sum(1), k_valid.sum(1)) <= self.sparse_threshold
            mode = "dense" if bool(use_dense.all()) else "probsparse" if bool((~use_dense).all()) else "mixed"
        self.last_mode = mode
        if mode == "dense":
            context = self._dense(q, k, v, k_valid)
        elif mode == "probsparse":
            context = self._probsparse(q, k, v, q_valid, k_valid)
        else:
            contexts = []
            for index in range(query.shape[0]):
                sample = slice(index, index + 1)
                if bool(use_dense[index]):
                    contexts.append(self._dense(q[sample], k[sample], v[sample], k_valid[sample]))
                else:
                    contexts.append(self._probsparse(q[sample], k[sample], v[sample], q_valid[sample], k_valid[sample]))
            context = torch.cat(contexts, dim=0)
        context = context.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.d_model)
        return zero_invalid(self.out_proj(context), q_valid)
