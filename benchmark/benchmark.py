"""Benchmark one segmentation model under the manuscript's inference setting.

This script measures parameter count, THOP operation counts, CUDA inference
latency, and peak allocated GPU memory for one model input. It does not train
the network and does not require a checkpoint for architecture-only measures.

``--size`` defaults to 224, the manuscript's slice-wise inference setting; pass
``--size 256`` to reproduce the complexity table's stated input size. The
manuscript's ``GFLOPs`` column corresponds to ``thop.thop_gmacs`` (one operation
per multiply-accumulate), not to ``thop.thop_gflops_2x``.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

from network import ENet, ODPNet, R2UNet, SegNet, UNet


def build_model(name: str, size: int = 224) -> torch.nn.Module:
    if name == "odpnet":
        return ODPNet(n_channels=3, n_classes=2, bilinear=False)
    if name == "unet":
        return UNet(n_channels=3, n_classes=2, bilinear=False)
    if name == "segnet":
        return SegNet(n_channels=3, n_classes=2)
    if name == "enet":
        return ENet(n_channels=3, n_classes=2)
    if name == "r2unet":
        # base_channels=32 reproduces the manuscript row (9.78 M params).
        # Use R2UNet(..., base_channels=64) in a standalone script for the
        # canonical upstream-width variant.
        return R2UNet(n_channels=3, n_classes=2, base_channels=32, t=2)
    if name == "mewunet":
        from network import MEWUNet

        return MEWUNet(n_channels=3, n_classes=2)
    # The baselines below depend on optional third-party packages (timm for
    # UNeXt and Polyp-PVT, torchvision for the PDAtt models), so they are
    # imported here instead of at module scope: a missing optional dependency
    # must not break benchmarking of the models that do not need it.
    if name == "unext":
        from network import UNext

        return UNext(n_channels=3, n_classes=2)
    if name == "pattunet":
        from network import PAttUNet

        return PAttUNet(n_channels=3, n_classes=2)
    if name == "dattunet":
        from network import DAttUNet

        return DAttUNet(n_channels=3, n_classes=2)
    if name == "polyp_pvt":
        from network import PolypPVT

        return PolypPVT(n_channels=3, n_classes=2)
    if name == "mdvit":
        from network import MDViT

        return MDViT(n_channels=3, n_classes=2, img_size=size)
    if name == "vm_unet":
        from network import VMUNet

        return VMUNet(n_channels=3, n_classes=2)
    if name == "ccvim":
        from network import CCViM

        return CCViM(n_channels=3, n_classes=2)
    if name == "vmamba":
        from network import VMambaClassifier

        return VMambaClassifier()
    raise ValueError(f"Unsupported model: {name}")


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
    state.pop("mask_values", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"checkpoint warning: missing={len(missing)}, unexpected={len(unexpected)}")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def thop_counts(
    model_factory: Callable[[], torch.nn.Module],
    x: torch.Tensor,
) -> dict[str, float] | None:
    try:
        from thop import profile
    except ImportError:
        print("THOP not installed; skipping operation count. Install with: python -m pip install thop")
        return None

    # THOP registers forward hooks and some versions may leave stale hooks on the
    # profiled module. Profile a temporary model so latency/memory tests use an
    # untouched CUDA model. Most models can be profiled on CPU, but VM-UNet and
    # other selective-scan models have a CUDA-only forward kernel.
    profile_model = model_factory().eval()
    cuda_only = getattr(profile_model, "benchmark_cuda_only", False)
    profile_x = torch.zeros(tuple(x.shape), dtype=x.dtype)
    try:
        if cuda_only:
            raise RuntimeError("model declares CUDA-only profiling")
        with torch.no_grad():
            macs, params = profile(profile_model, inputs=(profile_x,), verbose=False)
    except (RuntimeError, ValueError, NameError, ImportError) as exc:
        if not torch.cuda.is_available():
            print(f"THOP profile failed; skipping operation count: {exc}")
            return None
        print(f"CPU THOP profile failed ({exc}); retrying on CUDA.")
        del profile_model, profile_x
        torch.cuda.empty_cache()
        profile_model = model_factory().cuda().eval()
        profile_x = torch.zeros(tuple(x.shape), dtype=x.dtype, device="cuda")
        try:
            with torch.no_grad():
                macs, params = profile(profile_model, inputs=(profile_x,), verbose=False)
        except (RuntimeError, ValueError, NameError, ImportError) as cuda_exc:
            print(f"CUDA THOP profile failed; skipping operation count: {cuda_exc}")
            return None
    # THOP reports MACs although papers often label this column GFLOPs.
    return {
        "thop_macs": float(macs),
        "thop_gmacs": float(macs) / 1e9,
        "thop_gflops_2x": float(macs) * 2.0 / 1e9,
        "thop_params": float(params),
    }


def cuda_latency(model: torch.nn.Module, x: torch.Tensor, warmup: int, runs: int) -> dict[str, float]:
    # ``no_grad`` is intentional: some selective-scan autograd Functions reject
    # tensors created under ``inference_mode`` even during a forward-only pass.
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    with torch.no_grad():
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
    parser.add_argument(
        "--model",
        choices=(
            "odpnet",
            "unet",
            "segnet",
            "enet",
            "r2unet",
            "mewunet",
            "unext",
            "pattunet",
            "dattunet",
            "polyp_pvt",
            "mdvit",
            "vm_unet",
            "ccvim",
            "vmamba",
        ),
        default="odpnet",
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be positive")
    if args.model in {"mdvit", "vm_unet", "ccvim"} and args.size % 32:
        parser.error(f"{args.model} requires --size to be divisible by 32")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark must run on an NVIDIA GPU.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"input=({args.batch_size}, 3, {args.size}, {args.size}), dtype=float32")

    model_factory = lambda: build_model(args.model, args.size)
    model = model_factory().to(device).eval()
    load_checkpoint(model, str(args.checkpoint) if args.checkpoint else None)
    x = torch.randn(args.batch_size, 3, args.size, args.size, device=device, dtype=torch.float32)

    result: dict[str, object] = {
        "input_size": [args.batch_size, 3, args.size, args.size],
        "model": args.model,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "parameters": parameter_count(model),
        "parameters_million": parameter_count(model) / 1e6,
        "warmup": args.warmup,
        "runs": args.runs,
    }
    result["thop"] = thop_counts(model_factory, x)
    result["latency"] = cuda_latency(model, x, args.warmup, args.runs)
    result["peak_allocated_mb"] = peak_memory(model, x)

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
