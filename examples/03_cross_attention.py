"""Fuse two independent feature tensors with different temporal lengths."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from lanechange.components import LocalGlobalFusion

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(2)
    local_queries = torch.randn(2, 40, 64)
    global_context = torch.randn(2, 20, 64)
    fusion = LocalGlobalFusion(d_model=64, n_heads=4, mode="cross").eval()
    with torch.inference_mode():
        features, gate = fusion(local_queries, global_context)
    print("feature shape:", tuple(features.shape))
