"""Map a temporal feature sequence to direct future position offsets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from lanechange.components import TrajectoryPredictionHead

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(2)
    features = torch.randn(2, 40, 64)
    valid = torch.ones(2, 40, dtype=torch.bool)
    head = TrajectoryPredictionHead(d_model=64, horizon=40, output_dim=2).eval()
    with torch.inference_mode():
        offsets = head(features, valid)
    print("offset shape:", tuple(offsets.shape))
