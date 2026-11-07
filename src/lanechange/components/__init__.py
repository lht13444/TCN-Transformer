"""Temporal feature components and Transformer backbone."""

from .attention import ProjectedAttention
from .backbone import GlobalEncoder, GlobalEncoderLayer, HistoryEmbedding, SinusoidalPositionEncoding, TransformerBackbone
from .distillation import AttentionDistillation
from .fusion import LocalGlobalFusion
from .head import TrajectoryPredictionHead
from .masking import compact_padding, valid_mask, zero_invalid
from .tcn import CausalTemporalBlock, MultiScaleCausalTCN

__all__ = [
    "ProjectedAttention", "GlobalEncoder", "GlobalEncoderLayer",
    "HistoryEmbedding", "SinusoidalPositionEncoding", "TransformerBackbone",
    "AttentionDistillation", "LocalGlobalFusion", "compact_padding",
    "valid_mask", "zero_invalid", "CausalTemporalBlock", "MultiScaleCausalTCN",
    "TrajectoryPredictionHead",
]
