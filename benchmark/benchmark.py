"""Benchmark ODP-Net with the manuscript's 224x224 inference setting.

This script measures parameter count, THOP operation counts, CUDA inference
latency, and peak allocated GPU memory for one model input. It does not train
the network and does not require a checkpoint for architecture-only measures.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch

from network.main import ODPNet


def load_checkpoint(model: torch.nn.Module, checkpoint: str | None) -> None:
    if not checkpoint:
        return
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                state = raw[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint}")
    state = {str(k).removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"checkpoint warning: missing={len(missing)}, unexpected={len(unexpected)}")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def thop_counts(model: torch.nn.Module, x: torch.Tensor) -> dict[str, float] | None:
    try:
        from thop import profile
    except ImportError:
        print("THOP not installed; skipping operation count. Install with: python -m pip install thop")
        return None
    model.eval()
    with torch.inference_mode():
        macs, params = profile(model, inputs=(x,), verbose=False)
    # THOP reports MACs although papers often label this column GFLOPs.
    return {
        "thop_macs": float(macs),
        "thop_gmacs": float(macs) / 1e9,
        "thop_gflops_2x": float(macs) * 2.0 / 1e9,
        "thop_params": float(params),
    }


def cuda_latency(model: torch.nn.Module, x: torch.Tensor, warmup: int, runs: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    with torch.inference_mode():
        for start, end in zip(starts, ends):
            start.record()
            model(x)
            end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {
        "latency_ms_mean": statistics.mean(samples),
        "latency_ms_median": statistics.median(samples),
        "latency_ms_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "latency_ms_min": min(samples),
        "latency_ms_max": max(samples),
    }


def peak_memory(model: torch.nn.Module, x: torch.Tensor) -> float:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(x)
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark must run on an NVIDIA GPU.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"input=({args.batch_size}, 3, {args.size}, {args.size}), dtype=float32")

    model = ODPNet(n_channels=3, n_classes=2, bilinear=False).to(device).eval()
    load_checkpoint(model, str(args.checkpoint) if args.checkpoint else None)
    x = torch.randn(args.batch_size, 3, args.size, args.size, device=device, dtype=torch.float32)

    result: dict[str, object] = {
        "input_size": [args.batch_size, 3, args.size, args.size],
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "parameters": parameter_count(model),
        "parameters_million": parameter_count(model) / 1e6,
        "warmup": args.warmup,
        "runs": args.runs,
    }
    result["thop"] = thop_counts(model, x)
    result["latency"] = cuda_latency(model, x, args.warmup, args.runs)
    result["peak_allocated_mb"] = peak_memory(model, x)

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
