"""History features produced by the global temporal backbone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from lanechange.components import TransformerBackbone

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(2)
    history = torch.randn(2, 40, 6)
    valid = torch.ones(2, 40, dtype=torch.bool)
    backbone = TransformerBackbone(input_dim=6, d_model=64, attention="probsparse").eval()
    with torch.inference_mode():
        features, feature_valid = backbone(history, valid)
    print("feature shape:", tuple(features.shape))
    print("valid lengths:", feature_valid.sum(1).tolist())
    print("attention backends:", backbone.attention_modes)
