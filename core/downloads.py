"""Plugin-local TripoSplat weight download + progress state.

HF cache env vars are set in __init__.py BEFORE any huggingface_hub import.
We download into a flat ./models/ckpts/ tree so the pipeline ctor receives the
exact upstream subpaths (diffusion_models/..., vae/..., clip_vision/..., etc.).
"""
from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PLUGIN_DIR / "models"
CKPTS_DIR = MODELS_DIR / "ckpts"

REPO = "VAST-AI/TripoSplat"
APPROX_BYTES = 3_780_000_000  # ~3.78 GB, for progress estimation
REQUIRED_FILES = [
    "diffusion_models/triposplat_fp16.safetensors",
    "vae/triposplat_vae_decoder_fp16.safetensors",
    "clip_vision/dino_v3_vit_h.safetensors",
    "vae/flux2-vae.safetensors",
    "background_removal/birefnet.safetensors",
]

_lock = threading.Lock()
_state = {
    "stage": "idle",        # idle | checking | downloading | ready | error
    "progress": 0.0,
    "message": "",
    "error": "",
    "cancelled": False,
    "bytes_downloaded": 0,
    "bytes_total": 0,
}
_thread: threading.Thread | None = None


def _noop_log(_msg: str) -> None:
    return None


_log_fn: Callable[[str], None] = _noop_log


def get_state() -> dict:
    with _lock:
        return dict(_state)


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def _is_cancelled() -> bool:
    with _lock:
        return _state["cancelled"]


def set_logger(fn: Callable[[str], None]) -> None:
    global _log_fn
    _log_fn = fn


def ckpt_paths() -> dict:
    """The 5 explicit safetensors paths the pipeline ctor wants."""
    return {
        "ckpt_path": CKPTS_DIR / REQUIRED_FILES[0],
        "decoder_path": CKPTS_DIR / REQUIRED_FILES[1],
        "dinov3_path": CKPTS_DIR / REQUIRED_FILES[2],
        "flux2_vae_encoder_path": CKPTS_DIR / REQUIRED_FILES[3],
        "rmbg_path": CKPTS_DIR / REQUIRED_FILES[4],
    }


def is_weights_cached() -> bool:
    for rel in REQUIRED_FILES:
        p = CKPTS_DIR / rel
        if not (p.is_file() and p.stat().st_size > 1_000_000):
            return False
    return True


def is_ready() -> bool:
    return get_state()["stage"] == "ready" or is_weights_cached()


def start_background_download() -> None:
    """Kick off the download thread (no-op if running or already complete)."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        if _state["stage"] == "ready":
            return
    if is_weights_cached():
        _set(stage="ready", progress=1.0, message="Models ready")
        return
    _set(stage="checking", progress=0.0, message="Checking model cache...",
         error="", cancelled=False, bytes_downloaded=0, bytes_total=0)
    _thread = threading.Thread(target=_run, name="triposplat-model-dl", daemon=True)
    _thread.start()


def cancel_download() -> None:
    _set(cancelled=True)


def join(timeout: float = 2.0) -> None:
    t = _thread
    if t and t.is_alive():
        t.join(timeout=timeout)


def delete_models() -> None:
    cancel_download()
    join(timeout=3.0)
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR, ignore_errors=True)
    _set(stage="idle", progress=0.0, message="", error="", bytes_downloaded=0, bytes_total=0)


def _run() -> None:
    try:
        CKPTS_DIR.mkdir(parents=True, exist_ok=True)
        if is_weights_cached():
            _set(stage="ready", progress=1.0, message="Models ready")
            _log_fn("Weights already cached.")
            return
        _set(stage="downloading",
             message=f"Downloading TripoSplat weights (~{APPROX_BYTES // 1_000_000_000} GB)...",
             progress=0.01, bytes_total=APPROX_BYTES)
        _log_fn("Downloading TripoSplat weights to plugin-local cache...")
        _download()
        if _is_cancelled():
            _set(stage="error", error="Cancelled", message="Download cancelled")
            return
        _set(stage="ready", progress=1.0, message="Models ready")
        _log_fn("All weights cached in plugin dir.")
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        _log_fn(f"Model download failed: {msg}")
        _set(stage="error", error=msg, message="Download failed")


def _download() -> None:
    stop_flag = threading.Event()

    def watch():
        while not stop_flag.is_set():
            try:
                total = sum(f.stat().st_size for f in CKPTS_DIR.rglob("*") if f.is_file())
            except OSError:
                total = 0
            frac = min(0.99, total / APPROX_BYTES) if APPROX_BYTES > 0 else 0.0
            _set(progress=frac, bytes_downloaded=total, bytes_total=APPROX_BYTES)
            if _is_cancelled():
                return
            time.sleep(0.5)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=REPO,
            local_dir=str(CKPTS_DIR),
            allow_patterns=REQUIRED_FILES,
        )
    finally:
        stop_flag.set()
        watcher.join(timeout=1.0)
