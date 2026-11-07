# from __future__ import annotations
#
# import sys
# from dataclasses import dataclass
# from pathlib import Path
#
# sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# import torch
# from torch import Tensor, nn
# from lanechange.components import (
#     GlobalEncoder,
#     HistoryEmbedding,
#     LocalGlobalFusion,
#     MultiScaleCausalTCN,
#     TrajectoryPredictionHead,
#     zero_invalid,
# )
#
#
# @dataclass(frozen=True)
# class AssemblyConfig:
#     """Configure feature towers, repeated context fusion, and trajectory readout."""
#
#     input_dim: int = 6
#     d_model: int = 128
#     n_heads: int = 8
#     feedforward_dim: int = 512
#     local_depth: int = 2
#     local_dilations: tuple[int, ...] = (1, 2, 4, 8)
#     kernel_size: int = 3
#     global_depth: int = 3
#     attention: str = "probsparse"
#     sparse_factor: float = 3.0
#     distill: bool = True
#     fusion_depth: int = 2
#     fusion_mode: str = "cross"
#     head_hidden_dim: int = 256
#     horizon: int = 40
#     output_dim: int = 2
#     dropout: float = 0.1
#
#     def __post_init__(self) -> None:
#         dimensions = (
#             self.input_dim, self.d_model, self.n_heads,
#             self.feedforward_dim, self.local_depth, self.global_depth,
#             self.fusion_depth, self.head_hidden_dim, self.horizon,
#             self.output_dim, self.kernel_size,
#         )
#         if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in dimensions):
#             raise ValueError("Dimensions and block counts must be positive integers")
#         if self.d_model % self.n_heads:
#             raise ValueError("d_model must be divisible by n_heads")
#         if not 0 <= self.dropout < 1:
#             raise ValueError("dropout must be in [0, 1)")
#         if not self.local_dilations or any(rate < 1 for rate in self.local_dilations):
#             raise ValueError("local_dilations must contain positive rates")
#
#
# class TemporalResidualAdapter(nn.Module):
#     """Refine each feature token without mixing its temporal neighbors."""
#
#     def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
#         super().__init__()
#         self.norm = nn.LayerNorm(d_model)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(d_model, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, d_model),
#             nn.Dropout(dropout),
#         )
#
#     def forward(self, features: Tensor, valid: Tensor) -> Tensor:
#         features = zero_invalid(features, valid)
#         update = self.feed_forward(self.norm(features))
#         return zero_invalid(features + update, valid)
#
#
# class LocalFeatureStage(nn.Module):
#     """Combine multi-scale causal extraction with channel-wise refinement."""
#
#     def __init__(self, config: AssemblyConfig) -> None:
#         super().__init__()
#         self.temporal = MultiScaleCausalTCN(
#             d_model=config.d_model,
#             dilations=config.local_dilations,
#             kernel_size=config.kernel_size,
#             dropout=config.dropout,
#         )
#         self.refinement = TemporalResidualAdapter(
#             config.d_model, config.feedforward_dim, config.dropout,
#         )
#
#     def forward(self, features: Tensor, valid: Tensor) -> Tensor:
#         local = self.temporal(features, valid)
#         return self.refinement(local, valid)
#
#
# class LocalFeatureTower(nn.Module):
#     """Stack independently parameterized multi-scale temporal stages."""
#
#     def __init__(self, config: AssemblyConfig) -> None:
#         super().__init__()
#         self.stages = nn.ModuleList(
#             LocalFeatureStage(config) for _ in range(config.local_depth)
#         )
#         self.final_norm = nn.LayerNorm(config.d_model)
#
#     def forward(self, embedded: Tensor, valid: Tensor) -> Tensor:
#         features = embedded
#         for stage in self.stages:
#             features = stage(features, valid)
#         return zero_invalid(self.final_norm(features), valid)
#
#
# class ContextFusionStage(nn.Module):
#     """Refine local queries using a separately encoded global memory."""
#
#     def __init__(self, config: AssemblyConfig) -> None:
#         super().__init__()
#         self.query_norm = nn.LayerNorm(config.d_model)
#         self.memory_norm = nn.LayerNorm(config.d_model)
#         self.cross_fusion = LocalGlobalFusion(
#             d_model=config.d_model,
#             n_heads=config.n_heads,
#             dropout=config.dropout,
#             mode=config.fusion_mode,
#         )
#         self.refinement = TemporalResidualAdapter(
#             config.d_model, config.feedforward_dim, config.dropout,
#         )
#
#     def forward(
#         self,
#         local: Tensor,
#         memory: Tensor,
#         local_valid: Tensor,
#         memory_valid: Tensor,
#     ) -> Tensor:
#         queries = zero_invalid(self.query_norm(local), local_valid)
#         context = zero_invalid(self.memory_norm(memory), memory_valid)
#         fused, _ = self.cross_fusion(
#             queries, context, local_valid, memory_valid,
#         )
#         return self.refinement(fused, local_valid)
#
#
# class TCNTransformer(nn.Module):
#     """Assemble shared embedding, parallel towers, fusion, and a trajectory head."""
#
#     def __init__(self, config: AssemblyConfig | None = None) -> None:
#         super().__init__()
#         self.config = config if config is not None else AssemblyConfig()
#         cfg = self.config
#         self.history_embedding = HistoryEmbedding(
#             input_dim=cfg.input_dim,
#             d_model=cfg.d_model,
#             dropout=cfg.dropout,
#         )
#         self.local_tower = LocalFeatureTower(cfg)
#         self.global_tower = GlobalEncoder(
#             d_model=cfg.d_model,
#             n_heads=cfg.n_heads,
#             d_ff=cfg.feedforward_dim,
#             n_layers=cfg.global_depth,
#             dropout=cfg.dropout,
#             attention=cfg.attention,
#             sparse_factor=cfg.sparse_factor,
#             distill=cfg.distill,
#         )
#         self.context_stages = nn.ModuleList(
#             ContextFusionStage(cfg) for _ in range(cfg.fusion_depth)
#         )
#         self.fused_norm = nn.LayerNorm(cfg.d_model)
#         self.prediction_head = TrajectoryPredictionHead(
#             d_model=cfg.d_model,
#             horizon=cfg.horizon,
#             output_dim=cfg.output_dim,
#             hidden_dim=cfg.head_hidden_dim,
#             dropout=cfg.dropout,
#         )
#
#     def encode_history(
#         self,
#         history: Tensor,
#         time_valid: Tensor | None = None,
#     ) -> dict[str, Tensor]:
#         # Both towers process the same embedded observations in parallel.
#         embedded, local_valid = self.history_embedding(history, time_valid)
#         local_features = self.local_tower(embedded, local_valid)
#         global_features, global_valid = self.global_tower(embedded, local_valid)
#         return {
#             "local_features": local_features,
#             "global_features": global_features,
#             "local_valid": local_valid,
#             "global_valid": global_valid,
#         }
#
#     def fuse_context(self, encoded: dict[str, Tensor]) -> Tensor:
#         features = encoded["local_features"]
#         for stage in self.context_stages:
#             features = stage(
#                 features,
#                 encoded["global_features"],
#                 encoded["local_valid"],
#                 encoded["global_valid"],
#             )
#         return zero_invalid(self.fused_norm(features), encoded["local_valid"])
#
#     def forward(
#         self,
#         history: Tensor,
#         time_valid: Tensor | None = None,
#         return_features: bool = False,
#     ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
#         encoded = self.encode_history(history, time_valid)
#         fused = self.fuse_context(encoded)
#         # Each future point is a direct offset from the observation origin.
#         offsets = self.prediction_head(fused, encoded["local_valid"])
#         if return_features:
#             return offsets, {**encoded, "fused_features": fused}
#         return offsets
#
#
# if __name__ == "__main__":
#     torch.manual_seed(42)
#     torch.set_num_threads(2)
#     config = AssemblyConfig()
#     network = TCNTransformer(config).eval()
#     history = torch.randn(2, 40, config.input_dim)
#     time_valid = torch.ones(2, 40, dtype=torch.bool)
#     with torch.inference_mode():
#         offsets, features = network(history, time_valid, return_features=True)
#     print("offset shape:", tuple(offsets.shape))
#     print("local feature shape:", tuple(features["local_features"].shape))
#     print("global feature shape:", tuple(features["global_features"].shape))
