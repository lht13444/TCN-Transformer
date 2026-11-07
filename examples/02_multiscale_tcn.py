"""Inspect local receptive fields and differentiable scale fusion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from lanechange.components import MultiScaleCausalTCN

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(2)
    component = MultiScaleCausalTCN(d_model=64, dilations=(1, 2, 4), dropout=0)
    features = component(torch.randn(2, 40, 64))
    features[:, -1].square().mean().backward()
    print("feature shape:", tuple(features.shape))
    print("receptive fields:", component.receptive_fields)
    print("scale weights:", component.scale_weights.detach().tolist())
    print("scale gradients:", component.scale_logits.grad.tolist())
