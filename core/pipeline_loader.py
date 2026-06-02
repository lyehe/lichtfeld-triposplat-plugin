"""Cached, GPU-resident TripoSplatPipeline singleton (thread-locked)."""
from __future__ import annotations

import threading

from . import downloads

_lock = threading.Lock()
_pipeline = None  # type: ignore[var-annotated]


def is_loaded() -> bool:
    return _pipeline is not None


def _assert_cuda_and_bf16() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TripoSplat requires a CUDA GPU; no CUDA device is available. "
            "There is no CPU inference path."
        )
    major = torch.cuda.get_device_capability()[0]
    if major < 8:
        name = torch.cuda.get_device_name()
        raise RuntimeError(
            f"TripoSplat needs a GPU with compute capability >= 8.0 (bf16). "
            f"Detected '{name}' (sm_{major}x). Unsupported."
        )


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        if not downloads.is_weights_cached():
            raise RuntimeError("Model weights are not downloaded yet.")
        _assert_cuda_and_bf16()
        from triposplat import TripoSplatPipeline  # vendored, on sys.path
        paths = downloads.ckpt_paths()
        _pipeline = TripoSplatPipeline(
            ckpt_path=str(paths["ckpt_path"]),
            decoder_path=str(paths["decoder_path"]),
            dinov3_path=str(paths["dinov3_path"]),
            flux2_vae_encoder_path=str(paths["flux2_vae_encoder_path"]),
            rmbg_path=str(paths["rmbg_path"]),
            device="cuda",
        )
        return _pipeline


def unload() -> None:
    global _pipeline
    with _lock:
        _pipeline = None
    import gc
    for _ in range(2):
        gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
