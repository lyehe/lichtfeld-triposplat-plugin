"""TripoSplat — single-image 3D Gaussian splat generation for LichtFeld Studio."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# LFS embedded Python (Windows) hands a sys.stderr whose .flush() raises
# OSError(EINVAL); tqdm/HF call flush() on first import. Repoint to devnull.
if sys.stderr is not None:
    try:
        sys.stderr.flush()
    except OSError:
        sys.stderr = open(os.devnull, "w", buffering=1)

# Plugin-local HF cache — must be set BEFORE any huggingface_hub import.
_PLUGIN_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _PLUGIN_DIR / "models"
_HF_HOME = _MODELS_DIR / "huggingface"
_HF_HUB = _HF_HOME / "hub"
os.environ["HF_HOME"] = str(_HF_HOME)
os.environ["HF_HUB_CACHE"] = str(_HF_HUB)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_HF_HUB)

# Plugin-local torch.compile / Triton caches.
_COMPILE_CACHE = _PLUGIN_DIR / "cache" / "torch_compile"
_TRITON_CACHE = _PLUGIN_DIR / "cache" / "triton"
_COMPILE_CACHE.mkdir(parents=True, exist_ok=True)
_TRITON_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_COMPILE_CACHE))
os.environ.setdefault("TRITON_CACHE_DIR", str(_TRITON_CACHE))

# Expose the vendored `triposplat` package.
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

try:
    import lichtfeld as lf  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - test/DX path outside LFS host
    from types import ModuleType, SimpleNamespace

    class _Panel:
        pass

    lf = ModuleType("lichtfeld")
    lf.ui = SimpleNamespace(
        Panel=_Panel,
        PanelSpace=SimpleNamespace(MAIN_PANEL_TAB="MAIN_PANEL_TAB"),
        PanelHeightMode=SimpleNamespace(CONTENT="CONTENT"),
        free_plugin_textures=lambda _plugin_name: None,
        schedule_on_ui_thread=lambda fn: fn(),
    )
    lf.log = SimpleNamespace(
        info=lambda _msg: None,
        warn=lambda _msg: None,
        error=lambda _msg: None,
    )
    lf.register_class = lambda _cls: None
    lf.unregister_class = lambda _cls: None
    lf.stop_training = lambda: None
    sys.modules["lichtfeld"] = lf

from .core import downloads, pipeline_loader  # noqa: E402
from .panels.main_panel import TripoSplatPanel  # noqa: E402

_classes = [TripoSplatPanel]
_last_training_state = False


def _on_training_state_changed(new):
    global _last_training_state
    import threading
    is_now = bool(new)
    rising_edge = is_now and not _last_training_state
    _last_training_state = is_now
    if rising_edge and pipeline_loader.is_loaded():
        lf.log.info("[triposplat] Training started — unloading model to free VRAM.")
        threading.Thread(target=pipeline_loader.unload, daemon=True).start()


def _apply_perf_flags() -> None:
    try:
        import torch
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception as exc:  # noqa: BLE001
        lf.log.warn(f"[triposplat] couldn't apply perf flags: {exc}")


def _preload_models() -> None:
    """Download weights (if needed) and load the TripoSplat pipeline — which
    includes the RMBG/BiRefNet matting model — into VRAM at plugin load, so the
    Step 1 matte preview appears promptly on the first pasted/chosen image
    instead of waiting for a lazy first-use load. Runs on a daemon thread, and is
    skipped while training so it never competes with the trainer for VRAM."""
    try:
        from lfs_plugins.ui.state import AppState
        if AppState.is_training.value:
            lf.log.info("[triposplat] training active — deferring model preload.")
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        if not downloads.is_weights_cached():
            downloads.start_background_download()
            while not downloads.is_weights_cached():
                st = downloads.get_state()
                if st.get("cancelled") or st["stage"] == "error":
                    lf.log.warn("[triposplat] model preload aborted: weights unavailable.")
                    return
                downloads.join(timeout=1.0)  # wait on the download thread, re-check
        pipeline_loader.get_pipeline()  # loads diffusion model + RMBG/BiRefNet into VRAM
        lf.log.info("[triposplat] model + matting preloaded.")
    except Exception as exc:  # noqa: BLE001
        lf.log.warn(f"[triposplat] model preload failed: {exc}")


def on_load():
    downloads.set_logger(lambda msg: lf.log.info(f"[triposplat] {msg}"))
    _apply_perf_flags()
    for cls in _classes:
        lf.register_class(cls)
    try:
        global _last_training_state
        from lfs_plugins.ui.state import AppState
        _last_training_state = bool(AppState.is_training.value)
        AppState.is_training.subscribe_as("triposplat_plugin", _on_training_state_changed)
    except Exception as exc:  # noqa: BLE001
        lf.log.warn(f"triposplat_plugin: couldn't subscribe to is_training ({exc}).")
    # Preload weights + model (incl. RMBG/BiRefNet matting) on plugin load so the
    # first image's matte preview is ready immediately, not lazily on first use.
    import threading
    threading.Thread(target=_preload_models, name="triposplat-preload", daemon=True).start()
    lf.log.info("triposplat_plugin loaded")


def on_unload():
    import gc
    import time
    try:
        from lfs_plugins.ui.state import AppState
        if getattr(AppState, "is_training", None) is not None and AppState.is_training.value:
            try:
                lf.stop_training()
            except Exception:
                pass
            for _ in range(20):
                if not AppState.is_training.value:
                    break
                time.sleep(0.1)
    except Exception as exc:  # noqa: BLE001
        lf.log.warn(f"triposplat_plugin: stop_training on unload failed: {exc}")
    try:
        downloads.cancel_download()
        downloads.join(timeout=2.0)
    except Exception:
        pass
    try:
        pipeline_loader.unload()
    except Exception as exc:  # noqa: BLE001
        lf.log.warn(f"triposplat_plugin: pipeline_loader.unload() failed: {exc}")
    for _ in range(2):
        gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("triposplat_plugin unloaded")
