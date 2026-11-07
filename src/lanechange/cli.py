"""Runnable examples for individual temporal feature components."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .components import (
    AttentionDistillation, HistoryEmbedding, LocalGlobalFusion,
    MultiScaleCausalTCN, ProjectedAttention, TrajectoryPredictionHead, TransformerBackbone,
)
from .data import FEATURE_NAMES, get_observation_window, load_sample_trajectories

ROOT = Path(__file__).resolve().parents[2]


def tensor_shapes(value: Any) -> Any:
    """Describe component outputs without storing tensor values."""
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: tensor_shapes(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_shapes(item) for item in value]
    return value


def save_report(report: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def component_case(name: str, batch: int, steps: int, dim: int, device: str):
    """Create one independent feature-level example and its inputs."""
    x = torch.randn(batch, steps, dim, device=device)
    valid = torch.ones(batch, steps, dtype=torch.bool, device=device)
    if name == "tcn":
        module, inputs = MultiScaleCausalTCN(dim, (1, 2, 4), dropout=0), (x, valid)
    elif name == "attention":
        module = ProjectedAttention(dim, 4, dropout=0, mode="probsparse")
        inputs = (x, x, x, valid, valid)
    elif name == "distillation":
        module, inputs = AttentionDistillation(dim), (x, valid)
    elif name == "backbone":
        module = TransformerBackbone(6, dim, 4, 2 * dim, 2, dropout=0, attention="probsparse")
        history = torch.randn(batch, steps, 6, device=device)
        inputs = (history, valid)
    elif name == "fusion":
        # Independent features exercise the fusion interface directly.
        context = torch.randn(batch, max(1, steps // 2), dim, device=device)
        context_valid = torch.ones(context.shape[:2], dtype=torch.bool, device=device)
        module, inputs = LocalGlobalFusion(dim, 4, dropout=0, mode="cross"), (x, context, valid, context_valid)
    elif name == "prediction_head":
        module, inputs = TrajectoryPredictionHead(d_model=dim, horizon=40, dropout=0), (x, valid)
    else:
        raise ValueError(f"Unknown component: {name}")
    return module.to(device), inputs


def run_components(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    report: dict[str, Any] = {"example_seed": args.seed, "components": {}}
    for name in ["tcn", "attention", "distillation", "backbone", "fusion", "prediction_head"]:
        module, inputs = component_case(name, args.batch_size, args.steps, args.dim, args.device)
        module.eval()
        with torch.inference_mode():
            output = module(*inputs)
        details = {"inputs": tensor_shapes(inputs), "outputs": tensor_shapes(output)}
        if name == "tcn":
            details.update(receptive_fields=list(module.receptive_fields),
                           scale_weights=module.scale_weights.detach().cpu().tolist())
        if name == "attention":
            details["attention_mode"] = module.last_mode
        if name == "backbone":
            details["attention_modes"] = list(module.attention_modes)
        report["components"][name] = details
    sample_path, sample_format = args.sample, args.sample_format
    if sample_path is None:
        highd = sorted((ROOT / "data/highd_sample").glob("*_tracks.csv"))
        if highd:
            sample_path, sample_format = highd[0], "highd"
        else:
            sample_path, sample_format = ROOT / "data/raw_samples/ngsim_us101_12vehicles.csv", "ngsim"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Trajectory sample does not exist: {sample_path}")
    if sample_path.is_file():
        frame = load_sample_trajectories(sample_path, format=sample_format)
        sample = get_observation_window(frame, steps=args.steps, dt=.04)
        # A separate embedding example uses the real observation window.
        embedding = HistoryEmbedding(6, args.dim, dropout=0).to(args.device).eval()
        with torch.inference_mode():
            features, valid = embedding(torch.from_numpy(sample.x)[None].to(args.device))
        report["sample_embedding"] = {
            "source": {"ngsim": "NGSIM US-101", "highd": "highD", "canonical": "canonical trajectory sample"}[sample_format],
            "source_file": sample_path.name, "raw_rows": len(frame),
            "vehicle_groups": frame.groupby(["recording_id", "vehicle_id"]).ngroups,
            "feature_names": FEATURE_NAMES, "input_shape": list(sample.x.shape),
            "output_shape": list(features.shape), "valid_steps": int(valid.sum()),
            "observation_duration_s": float(sample.time_s[-1] - sample.time_s[0]),
        }
    save_report(report, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)
    components = sub.add_parser("components", help="Run each independent component")
    components.add_argument("--sample", type=Path)
    components.add_argument("--sample-format", choices=["highd", "ngsim", "canonical"], default="highd")
    components.add_argument("--batch-size", type=int, default=2)
    components.add_argument("--steps", type=int, default=40)
    components.add_argument("--dim", type=int, default=64)
    components.add_argument("--device", default="cpu")
    components.add_argument("--seed", type=int, default=42)
    components.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if min(args.batch_size, args.steps, args.dim) < 1 or args.dim % 4:
        parser.error("positive shapes required; dim must be divisible by four")
    run_components(args)


if __name__ == "__main__":
    main()
