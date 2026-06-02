# TripoSplat Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LichtFeld Studio plugin that turns a single image into a 3D Gaussian splat (via VAST-AI TripoSplat), with a preprocessed-matte preview, generation settings, instant re-decode at different densities, and interactive placement of the result into the current (possibly existing) scene.

**Architecture:** A faithful mirror of the HYWorld2 / VGGT-Omega plugins — vendored upstream package on `sys.path`, plugin-local HF cache, a cached GPU-resident pipeline singleton, a daemon `Job` that streams staged progress, and a hybrid RML/RCSS panel. The decoded `Gaussian` is inserted into the scene by writing it to a tiny temp `.ply` (`Gaussian.to_ply_bytes()`, which already applies the axis-remap + log-scale + logit-opacity) and loading it with `lf.io.load` + `scene.add_splat` (the ml-sharp pattern). All scene mutation is marshaled to the UI thread.

**Tech Stack:** Python 3.12, torch 2.11.0 (cu130), huggingface-hub, vendored `triposplat` (torch/torchvision/numpy/safetensors/pillow/tqdm), LichtFeld Studio plugin API (`lichtfeld` / `lfs_plugins`), RmlUi data-binding, pytest + ruff + uv.

---

## ⚠️ Deviation from the approved spec (read first)

The spec chose **in-memory DLPack capture** as the primary insertion path with manual activation-inversion + axis-remap, and PLY export only for "Save to disk". Having since read the real upstream source, this plan uses **PLY round-trip as the primary insertion path** instead:

- `Gaussian.to_ply_bytes(transform=None)` already applies `_DEFAULT_TRANSFORM` (Y↔Z axis remap), writes **log-scale**, **logit-opacity**, and correct **wxyz** quaternions — i.e. it reproduces, with upstream's own tested code, exactly the math the spec wanted us to hand-port.
- We write those bytes to a temp `.ply` (~18 MB for 262 144 gaussians — trivial) and load via `lf.io.load(path).splat_data` → `scene.add_splat(..., *_raw ...)` — the **exact pattern ml-sharp ships** for single-image→splat.
- This **eliminates the spec's #1 risk** (wrong quaternion/axis-remap math) and removes the DLPack-ownership trap, at a negligible I/O cost.
- "Save to disk" reuses the same `save_ply` / `save_splat` methods.

If you prefer the DLPack path, Task 6 documents it as the alternative. Everything else in the spec stands.

**Other ground-truth corrections folded in:** the preprocessed matte is **RGB** (composited on black), not RGBA; the pipeline ctor takes **5 explicit safetensors paths**; `run()`/method defaults are seed=42, steps=20, guidance=3.0, shift=3.0, num_gaussians=262144, erode_radius=1; gaussian count is clamped to [32768, 262144] and rounded to a multiple of 32; `decode_latent(latent, num_gaussians)` returns a single `Gaussian` and is the cheap re-decode entry; `sample_latent(cond, steps, guidance_scale, shift, generator, show_progress, callback)` accepts a per-step `callback` we use for progress.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Manifest: cu130 torch pins, uv index/sources, ruff, pytest, `[tool.lichtfeld]`. |
| `.gitignore` | Root-anchored ignores (`/models/`, `/cache/`, `/.venv/`, `/uv.lock`). |
| `LICENSE`, `NOTICE` | MIT; vendored-code + weight attributions. |
| `conftest.py` | Lets pytest collect without importing the host-only root `__init__.py`. |
| `install.ps1` / `.sh`, `uninstall.ps1` / `.sh` | Junction/symlink into `~/.lichtfeld/plugins/triposplat_plugin`; cache cleanup. |
| `.github/workflows/ci.yml` | ruff + structure check + `uv lock` on ubuntu+windows. |
| `__init__.py` | Lifecycle: stderr guard → HF cache env → sys.path → register → bg download → training subscribe → teardown. |
| `triposplat/` | Vendored upstream (`triposplat.py`, `model.py`, …) at a pinned commit; on `sys.path`. |
| `core/downloads.py` | Background weight download of `VAST-AI/TripoSplat` (5 files) into `models/ckpts/`; progress/cancel/`is_ready`. |
| `core/pipeline_loader.py` | Cached GPU `TripoSplatPipeline` singleton; bf16/compute guard; load/unload. |
| `core/preprocess.py` | `preprocess_image` wrapper → RGB matte (PIL + tensor for preview). |
| `core/job.py` | `JobConfig`/`JobResult` + daemon `TripoSplatJob`; full run + `decode_only`; latent cache. |
| `core/insertion.py` | Gaussian → temp `.ply` → `lf.io.load` → `scene.add_splat`; replace/append node naming. |
| `core/export.py` | `save_ply` / `save_splat` to a user path (Save to disk). |
| `panels/main_panel.py` + `.rml` + `.rcss` | Hybrid panel: input, matte preview, settings, model, placement (T/R/S + gizmo), run/cancel, result. |
| `tests/` | GPU-free unit tests: downloads completeness, job latent-cache, append-name, num_gaussians rounding. |

---

## Conventions used by every task

- Run all Python via `uv run` (per user global rule).
- Work from the plugin root: `C:/Users/HEQ/Projects/gaussian_splatting/plugins/lichtfeld-triposplat-plugin`.
- Sibling reference paths (read these for verbatim copies):
  - HYWorld2 root: `C:/Users/HEQ/Projects/gaussian_splatting/plugins/Lichtfeld-HYWorld2-Plugin`
  - VGGT-Omega root: `C:/Users/HEQ/Projects/gaussian_splatting/plugins/lichtfeld-vggt-omega`
  - ml-sharp (insertion-via-lf.io.load reference): `C:/Users/HEQ/Projects/gaussian_splatting/plugins/Lichtfeld-ml-sharp-Plugin`
- "Copy verbatim from `<path>`, apply edits" means the engineer opens that on-disk file and reproduces it with the listed changes — the file exists locally.
- `# verify:` notes mark a **host API contract** (a `lichtfeld`/`lfs_plugins` symbol) to confirm against the LFS stubs at `C:/Users/HEQ/Projects/gaussian_splatting/LichtFeld-Studio/src/python/stubs/` or against the cited sibling before relying on it. These are external APIs, not gaps in our code.

---

## Task 0: Initialize git + scaffold directories

**Files:**
- Create: repo at plugin root; empty package dirs.

- [ ] **Step 1: Init the repo and package dirs**

```bash
cd "C:/Users/HEQ/Projects/gaussian_splatting/plugins/lichtfeld-triposplat-plugin"
git init
mkdir -p core panels tests triposplat .github/workflows docs/superpowers/specs docs/superpowers/plans
```

- [ ] **Step 2: Create empty package markers**

Create `core/__init__.py`, `panels/__init__.py`, and `tests/__init__.py` as empty files. Create `triposplat/__init__.py` as empty for now (replaced in Task 2).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: scaffold triposplat plugin directories"
```

---

## Task 1: Packaging files (pyproject, gitignore, license, conftest, CI)

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE`, `NOTICE`, `conftest.py`, `.github/workflows/ci.yml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "TripoSplat"
version = "0.1.0"
description = "Generate a 3D Gaussian splat from a single image using VAST-AI TripoSplat, inserted directly into the scene."
requires-python = ">=3.12,<3.13"
authors = [{ name = "Yehe Liu" }]

dependencies = [
    # `triposplat` is vendored inline at ./triposplat/ and put on sys.path by __init__.py.
    "torch==2.11.0",
    "torchvision==0.26.0",
    "numpy",
    "safetensors",
    "pillow",
    "tqdm",
    "huggingface-hub",
    "triton-windows>=3.5 ; platform_system == 'Windows'",
]

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.ruff]
extend-exclude = ["triposplat", ".venv", "cache", "models"]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
ignore = ["E501"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "--import-mode=importlib"

[tool.lichtfeld]
hot_reload = false
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.5.0"
required_features = []
auto_start = false
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
/.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
*.egg-info/
/uv.lock
# Root-anchored: only top-level downloaded weights + JIT caches are ignored.
/models/
/cache/
```

- [ ] **Step 3: Write `conftest.py`** — copy verbatim from `lichtfeld-vggt-omega/conftest.py` (it keys off `__file__`'s parent, nothing to swap):

```python
# conftest.py — pytest configuration
#
# The plugin root has an __init__.py required by LichtFeld Studio, but that
# file imports `lichtfeld` (only available inside the LFS host). pytest 9.x
# treats the root as a Package and Package.setup() unconditionally calls
# importtestmodule(__init__.py), which fails outside LFS.
#
# Fix: patch Package.setup() so it skips the root __init__.py only.
from __future__ import annotations

from pathlib import Path

from _pytest.python import Package

_ROOT = Path(__file__).resolve().parent
_original_setup = Package.setup


def _patched_setup(self: Package) -> None:
    if self.path.resolve() == _ROOT:
        return  # skip importtestmodule(__init__.py) for the plugin root
    _original_setup(self)


Package.setup = _patched_setup  # type: ignore[method-assign]
```

- [ ] **Step 4: Write `LICENSE`** — standard MIT text, `Copyright (c) 2026 Yehe Liu`.

- [ ] **Step 5: Write `NOTICE`**

```
TripoSplat — LichtFeld Studio plugin

This plugin vendors the VAST-AI-Research/TripoSplat source (MIT License) under
./triposplat/. Model weights (HuggingFace repo VAST-AI/TripoSplat, MIT License,
~3.78 GB across 5 safetensors files) are downloaded at runtime into ./models/
and are NOT redistributed with this plugin.

Bundled/runtime dependencies and their licenses:
  - torch, torchvision         (BSD-3-Clause)
  - numpy                      (BSD-3-Clause)
  - safetensors                (Apache-2.0)
  - pillow                     (HPND)
  - tqdm                       (MPL-2.0 / MIT)
  - huggingface-hub            (Apache-2.0)
```

- [ ] **Step 6: Write `.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]
jobs:
  check:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Structure integrity
        run: |
          test -f triposplat/triposplat.py
          test -f __init__.py
          test -f pyproject.toml
        shell: bash
      - name: Ruff
        run: uvx ruff check .
      - name: Lock resolves
        run: uv lock --no-cache
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore LICENSE NOTICE conftest.py .github/workflows/ci.yml
git commit -m "build: packaging, license, conftest, CI"
```

---

## Task 2: Vendor upstream TripoSplat + verify the API

**Files:**
- Create: `triposplat/` (downloaded upstream source), `triposplat/PIN.txt`

- [ ] **Step 1: Download the upstream source into `triposplat/`**

Fetch the repository at a pinned commit. Replace `<COMMIT>` with the resolved HEAD of `main` at vendoring time.

```bash
cd "C:/Users/HEQ/Projects/gaussian_splatting/plugins/lichtfeld-triposplat-plugin"
git clone https://github.com/VAST-AI-Research/TripoSplat _ts_tmp
cd _ts_tmp && git rev-parse HEAD > ../triposplat/PIN.txt && cd ..
# Copy the python package files (triposplat.py, model.py, __init__.py, and any local modules they import)
cp _ts_tmp/triposplat.py _ts_tmp/model.py triposplat/
# If upstream ships an __init__.py exporting TripoSplatPipeline, copy it; else create one (Step 2).
cp _ts_tmp/__init__.py triposplat/ 2>/dev/null || true
rm -rf _ts_tmp
```

- [ ] **Step 2: Ensure `triposplat/__init__.py` exports the public API**

If upstream has no package `__init__.py`, create `triposplat/__init__.py`:

```python
from .triposplat import TripoSplatPipeline, Gaussian  # noqa: F401
```

- [ ] **Step 3: Verify the API against the ground-truth contract**

Read `triposplat/triposplat.py` and confirm each of these exists with the expected shape (this is the authoritative checklist; if anything differs, update the names used in Tasks 4–6 to match the real source):

```
class TripoSplatPipeline:
    def __init__(self, ckpt_path, decoder_path, dinov3_path,
                 flux2_vae_encoder_path, rmbg_path, device="cuda")
    def preprocess_image(self, image, erode_radius=1) -> PIL.Image   # RGB, 1024x1024, on black
    def encode_image(self, image, generator=None) -> dict            # {'feature1','feature2'}
    def sample_latent(self, cond, steps=50, guidance_scale=7.0, shift=3.0,
                      generator=None, show_progress=False, callback=None) -> dict  # has 'latent'
    def decode_latent(self, latent, num_gaussians=262144) -> Gaussian
    @torch.no_grad()
    def run(self, image, seed=42, steps=20, guidance_scale=3.0, shift=3.0,
            num_gaussians=262144, erode_radius=1, show_progress=False, callback=None)
    def _validate_num_gaussians(self, n) -> int                      # [32768,262144], mult of 32

class Gaussian:
    _DEFAULT_TRANSFORM = [[1,0,0],[0,0,-1],[0,1,0]]
    def save_ply(self, path, transform=None)
    def save_splat(self, path, transform=None)
    def to_ply_bytes(self, transform=None) -> bytes                  # applies _DEFAULT_TRANSFORM when None
    def to_splat_bytes(self, transform=None) -> bytes
    # accessors: get_xyz, get_scaling, get_opacity, _features_dc, _rotation, rots_bias, sh_degree
```

Confirm the 5 weight subpaths used by the ctor in `run_example.py` / `run_gradio.py`:
`diffusion_models/triposplat_fp16.safetensors`, `vae/triposplat_vae_decoder_fp16.safetensors`, `clip_vision/dino_v3_vit_h.safetensors`, `vae/flux2-vae.safetensors`, `background_removal/birefnet.safetensors`.

- [ ] **Step 4: Sanity import (no GPU needed for import)**

```bash
uv run python -c "import sys; sys.path.insert(0,'.'); from triposplat import TripoSplatPipeline, Gaussian; print('ok', Gaussian._DEFAULT_TRANSFORM)"
```
Expected: `ok [[1, 0, 0], [0, 0, -1], [0, 1, 0]]` (a clean import; if it pulls heavy deps, that's fine once `uv sync` has run).

- [ ] **Step 5: Commit**

```bash
git add triposplat
git commit -m "vendor: TripoSplat upstream source (pinned commit)"
```

---

## Task 3: `core/downloads.py` (weight downloader) + tests

**Files:**
- Create: `core/downloads.py`, `tests/test_downloads.py`

- [ ] **Step 1: Write the failing test `tests/test_downloads.py`**

```python
from pathlib import Path

from core import downloads


def test_required_files_list():
    # Exactly the 5 upstream weight files, with their subdirs.
    assert downloads.REQUIRED_FILES == [
        "diffusion_models/triposplat_fp16.safetensors",
        "vae/triposplat_vae_decoder_fp16.safetensors",
        "clip_vision/dino_v3_vit_h.safetensors",
        "vae/flux2-vae.safetensors",
        "background_removal/birefnet.safetensors",
    ]


def test_is_weights_cached_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "CKPTS_DIR", tmp_path)
    assert downloads.is_weights_cached() is False


def test_is_weights_cached_true_when_all_present(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "CKPTS_DIR", tmp_path)
    for rel in downloads.REQUIRED_FILES:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 2_000_000)  # > 1 MB sanity threshold
    assert downloads.is_weights_cached() is True


def test_ckpt_paths_returns_five_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "CKPTS_DIR", tmp_path)
    paths = downloads.ckpt_paths()
    assert set(paths) == {"ckpt_path", "decoder_path", "dinov3_path",
                          "flux2_vae_encoder_path", "rmbg_path"}
    assert paths["ckpt_path"] == tmp_path / "diffusion_models/triposplat_fp16.safetensors"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_downloads.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: module 'core.downloads' has no attribute 'REQUIRED_FILES'`.

- [ ] **Step 3: Write `core/downloads.py`**

Adapted from `Lichtfeld-HYWorld2-Plugin/core/downloads.py` (same thread-safe state machine + disk-watcher progress thread), re-pointed to a single ungated repo downloaded into a **flat `models/ckpts/` layout** (so the pipeline ctor gets the exact upstream subpaths), and a 5-file completeness check:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_downloads.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add core/downloads.py tests/test_downloads.py
git commit -m "feat(core): TripoSplat weight downloader with completeness check"
```

---

## Task 4: `core/pipeline_loader.py` (cached singleton + bf16 guard)

**Files:**
- Create: `core/pipeline_loader.py`

- [ ] **Step 1: Write `core/pipeline_loader.py`**

```python
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
```

- [ ] **Step 2: Structure check (import without GPU)**

Run: `uv run python -c "import sys; sys.path.insert(0,'.'); from core import pipeline_loader; print(pipeline_loader.is_loaded())"`
Expected: `False` (module imports cleanly; no pipeline built).

- [ ] **Step 3: Commit**

```bash
git add core/pipeline_loader.py
git commit -m "feat(core): cached TripoSplatPipeline loader with bf16/compute guard"
```

---

## Task 5: `core/preprocess.py` and `core/export.py`

**Files:**
- Create: `core/preprocess.py`, `core/export.py`

- [ ] **Step 1: Write `core/preprocess.py`**

```python
"""Wrapper around TripoSplatPipeline.preprocess_image.

Returns the upstream RGB matte (1024x1024, background removed, composited on
black) as a PIL image plus a contiguous uint8 [H, W, 3] numpy array suitable
for the panel's ui.image_tensor preview.
"""
from __future__ import annotations


def run_preprocess(pipeline, pil_image, erode_radius: int = 1):
    """-> (matte_pil: PIL.Image RGB, matte_rgb: np.ndarray[H,W,3] uint8)."""
    import numpy as np

    matte_pil = pipeline.preprocess_image(pil_image, erode_radius=erode_radius)
    if matte_pil.mode != "RGB":
        matte_pil = matte_pil.convert("RGB")
    matte_rgb = np.ascontiguousarray(np.asarray(matte_pil, dtype=np.uint8))
    return matte_pil, matte_rgb
```

- [ ] **Step 2: Write `core/export.py`**

```python
"""Save a decoded Gaussian to disk (Save-to-disk action).

Uses upstream's own writers, which apply Gaussian._DEFAULT_TRANSFORM by default
(transform=None) — identical orientation/scale/opacity convention as the splat
we inserted into the scene.
"""
from __future__ import annotations

from pathlib import Path


def save(gaussian, path: str, fmt: str = "ply") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "ply":
        gaussian.save_ply(str(p))
    elif fmt == "splat":
        gaussian.save_splat(str(p))
    else:
        raise ValueError(f"Unknown format: {fmt!r} (expected 'ply' or 'splat')")
    return str(p)
```

- [ ] **Step 3: Import check**

Run: `uv run python -c "import sys; sys.path.insert(0,'.'); from core import preprocess, export; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add core/preprocess.py core/export.py
git commit -m "feat(core): preprocess matte wrapper + disk export"
```

---

## Task 6: `core/insertion.py` (Gaussian → temp .ply → scene) + append-name test

**Files:**
- Create: `core/insertion.py`, `tests/test_insertion.py`

- [ ] **Step 1: Write the failing test `tests/test_insertion.py`** (covers the GPU-free append-name logic)

```python
from core import insertion


class _FakeScene:
    def __init__(self, existing):
        self._names = set(existing)

    def has_node(self, name):
        return name in self._names


def test_next_group_name_replace_mode_returns_base():
    scene = _FakeScene({"TripoSplat", "TripoSplat_01"})
    assert insertion.next_group_name(scene, append=False) == "TripoSplat"


def test_next_group_name_append_picks_first_free_slot():
    scene = _FakeScene({"TripoSplat", "TripoSplat_01"})
    assert insertion.next_group_name(scene, append=True) == "TripoSplat_02"


def test_next_group_name_append_from_empty_scene():
    scene = _FakeScene(set())
    assert insertion.next_group_name(scene, append=True) == "TripoSplat_01"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_insertion.py -v`
Expected: FAIL (`AttributeError: module 'core.insertion' has no attribute 'next_group_name'`).

- [ ] **Step 3: Write `core/insertion.py`**

```python
"""Insert a decoded Gaussian into the current LFS scene.

Primary path (this file): write Gaussian.to_ply_bytes() to a temp .ply and load
it with lf.io.load(...).splat_data, then scene.add_splat(...) using the *_raw
fields (the ml-sharp pattern). to_ply_bytes() already applies the axis remap,
log-scale, logit-opacity, and wxyz quaternions, so no manual math here.

MUST run on the UI thread (call from inside lf.ui.schedule_on_ui_thread).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

BASE_NAME = "TripoSplat"


def next_group_name(scene, append: bool) -> str:
    """Replace mode -> the base name. Append mode -> first free TripoSplat_NN."""
    if not append:
        return BASE_NAME
    i = 1
    while True:
        candidate = f"{BASE_NAME}_{i:02d}"
        # verify: Scene.has_node — confirm the host exposes a name lookup; if not,
        # track inserted names in the panel and pass an `existing` set instead.
        if not scene.has_node(candidate):
            return candidate
        i += 1


def insert_gaussian(gaussian, *, append: bool, log=None) -> str | None:
    """Insert into lf.get_scene(); returns the splat node name or None."""
    import lichtfeld as lf

    _log = log or (lambda _m: None)
    scene = lf.get_scene()
    if scene is None or not scene.is_valid():
        _log("insertion: no valid scene; skipping.")
        return None

    group_name = next_group_name(scene, append=append)
    if not append:
        try:
            scene.remove_node(group_name, keep_children=False)
        except Exception:
            pass

    parent_id = scene.add_group(group_name)
    splat_name = f"{group_name} / splats"

    # Write the upstream PLY (default transform = axis remap applied) to a temp file.
    with tempfile.TemporaryDirectory() as td:
        ply_path = Path(td) / "triposplat.ply"
        ply_path.write_bytes(gaussian.to_ply_bytes())
        # verify: lf.io.load(...).splat_data and *_raw field names confirmed in
        # Lichtfeld-ml-sharp-Plugin (sharp_processor.py / panels). Match its usage.
        result = lf.io.load(str(ply_path))
        sd = result.splat_data
        scene.add_splat(
            name=splat_name,
            means=sd.means_raw,
            sh0=sd.sh0_raw,
            shN=sd.shN_raw,
            scaling=sd.scaling_raw,
            rotation=sd.rotation_raw,
            opacity=sd.opacity_raw,
            sh_degree=sd.active_sh_degree,
            scene_scale=sd.scene_scale,
            parent=parent_id,
        )
    try:
        scene.notify_changed()
    except Exception as exc:  # noqa: BLE001
        _log(f"insertion: notify_changed failed: {exc}")
    _log(f"insertion: added '{splat_name}'.")
    return splat_name
```

> **Alternative (DLPack, per original spec):** replace the temp-PLY block with `_to_lf`-wrapped tensors captured directly from the Gaussian — `means = gaussian.get_xyz @ T.T`, `scaling = torch.log(gaussian.get_scaling.clamp(min=1e-8))`, `opacity = torch.logit(gaussian.get_opacity.clamp(1e-6,1-1e-6))`, `sh0 = gaussian._features_dc`, `shN = zeros[N,0,3]`, and the quaternion rotated by `T` using the vendored `triposplat._quat_to_matrix`/`_matrix_to_quat`. Keep the source tensors alive for the node's lifetime. Only adopt this if profiling shows the temp-PLY I/O matters (it won't at ≤262k gaussians).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_insertion.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add core/insertion.py tests/test_insertion.py
git commit -m "feat(core): scene insertion via temp PLY + lf.io.load (ml-sharp pattern)"
```

---

## Task 7: `core/job.py` (daemon job + latent cache) + cache tests

**Files:**
- Create: `core/job.py`, `tests/test_job_cache.py`

- [ ] **Step 1: Write the failing test `tests/test_job_cache.py`**

```python
from core.job import JobConfig, latent_signature, num_gaussians_valid


def _cfg(**kw):
    base = dict(image_path="a.png", seed=42, steps=20, guidance_scale=3.0,
               shift=3.0, num_gaussians=262144, erode_radius=1, append=False)
    base.update(kw)
    return JobConfig(**base)


def test_jobconfig_is_frozen():
    cfg = _cfg()
    try:
        cfg.seed = 7  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("JobConfig must be frozen")


def test_signature_changes_when_sampling_inputs_change():
    base = latent_signature(_cfg(), preprocessed_id="m1")
    assert latent_signature(_cfg(seed=43), preprocessed_id="m1") != base
    assert latent_signature(_cfg(steps=30), preprocessed_id="m1") != base
    assert latent_signature(_cfg(guidance_scale=4.0), preprocessed_id="m1") != base
    assert latent_signature(_cfg(shift=2.0), preprocessed_id="m1") != base
    assert latent_signature(_cfg(), preprocessed_id="m2") != base


def test_signature_unchanged_when_only_count_changes():
    base = latent_signature(_cfg(num_gaussians=262144), preprocessed_id="m1")
    assert latent_signature(_cfg(num_gaussians=65536), preprocessed_id="m1") == base


def test_num_gaussians_validation_rounds_to_multiple_of_32():
    assert num_gaussians_valid(262144) == 262144
    assert num_gaussians_valid(65536) == 65536
    assert num_gaussians_valid(100000) == 100000 // 32 * 32 + (32 if (100000 % 32) >= 16 else 0)
    assert num_gaussians_valid(10) == 32768       # clamped up to min
    assert num_gaussians_valid(999999) == 262144  # clamped down to max
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_job_cache.py -v`
Expected: FAIL (`ImportError`/`AttributeError`).

- [ ] **Step 3: Write `core/job.py`**

Threading skeleton copied from `lichtfeld-vggt-omega/core/job.py` (JobStage enum, lock-guarded props, deque log, `_check_cancel`, `start`/`cancel`, `_run` try/except→failed-result), with TripoSplat's `JobConfig`/`JobResult`, the staged body, the latent cache, and the `decode_only` entry.

```python
from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum

import lichtfeld as lf

from . import insertion, pipeline_loader, preprocess

_NUM_GAUSSIANS_MIN = 32768
_NUM_GAUSSIANS_MAX = 262144
_GAUSSIANS_PER_POINT = 32


def num_gaussians_valid(n: int) -> int:
    n = max(_NUM_GAUSSIANS_MIN, min(_NUM_GAUSSIANS_MAX, int(n)))
    if n % _GAUSSIANS_PER_POINT != 0:
        n = round(n / _GAUSSIANS_PER_POINT) * _GAUSSIANS_PER_POINT
    return max(_NUM_GAUSSIANS_MIN, min(_NUM_GAUSSIANS_MAX, n))


class JobStage(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    LOADING_MODEL = "loading_model"
    PREPROCESS = "preprocess"
    ENCODE = "encode"
    SAMPLE = "sample"
    DECODE = "decode"
    INSERTING = "inserting"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


_RUNNING = {JobStage.PREPARING, JobStage.LOADING_MODEL, JobStage.PREPROCESS,
            JobStage.ENCODE, JobStage.SAMPLE, JobStage.DECODE, JobStage.INSERTING}


@dataclass(frozen=True)
class JobConfig:
    image_path: str
    seed: int = 42
    steps: int = 20
    guidance_scale: float = 3.0
    shift: float = 3.0
    num_gaussians: int = 262144
    erode_radius: int = 1
    append: bool = False


@dataclass
class JobResult:
    success: bool
    elapsed_s: float = 0.0
    num_gaussians: int = 0
    node_name: str = ""
    error: str = ""


def latent_signature(cfg: JobConfig, preprocessed_id: str) -> tuple:
    """Identifies a sampled latent. num_gaussians is deliberately excluded."""
    return (cfg.seed, cfg.steps, round(cfg.guidance_scale, 4),
            round(cfg.shift, 4), preprocessed_id)


class _Cancelled(Exception):
    pass


class TripoSplatJob:
    def __init__(self, cfg: JobConfig, *, cached_latent=None, cached_signature=None,
                 matte_pil=None, preprocessed_id: str = ""):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancelled = False
        self._stage = JobStage.IDLE
        self._progress = 0.0
        self._status = ""
        self._result: JobResult | None = None
        self._log: deque[str] = deque(maxlen=48)
        # latent cache carried across jobs by the panel:
        self._latent = cached_latent
        self._signature = cached_signature
        self._matte_pil = matte_pil
        self._preprocessed_id = preprocessed_id
        self._gaussian = None  # kept alive for the node lifetime

    # ---- lock-guarded props (verbatim shape from vggt-omega) ----
    @property
    def stage(self):
        with self._lock:
            return self._stage

    @property
    def progress(self):
        with self._lock:
            return self._progress

    @property
    def status(self):
        with self._lock:
            return self._status

    @property
    def result(self):
        with self._lock:
            return self._result

    @property
    def log_text(self):
        with self._lock:
            return "\n".join(self._log)

    @property
    def latent(self):
        return self._latent

    @property
    def signature(self):
        return self._signature

    @property
    def gaussian(self):
        return self._gaussian

    def is_running(self) -> bool:
        return self.stage in _RUNNING

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("Job already started")
            self._thread = threading.Thread(target=self._run, daemon=True)
            thread = self._thread
        thread.start()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            self._status = "Cancelling..."

    def _set(self, stage, progress, status):
        with self._lock:
            self._stage, self._progress, self._status = stage, progress, status

    def _log_line(self, msg):
        line = str(msg).rstrip()
        if not line:
            return
        with self._lock:
            self._log.append(line)
        lf.log.info(f"[triposplat] {line}")

    def _is_cancelled(self):
        with self._lock:
            return self._cancelled

    def _check_cancel(self):
        if self._is_cancelled():
            raise _Cancelled()

    def _run(self):
        t0 = time.time()
        try:
            self._run_pipeline(t0)
        except _Cancelled:
            with self._lock:
                progress = self._progress
                self._result = JobResult(False, error="Cancelled", elapsed_s=time.time() - t0)
            self._set(JobStage.CANCELLED, progress, "Cancelled")
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            self._log_line(f"ERROR: {msg}")
            self._log_line(traceback.format_exc())
            with self._lock:
                progress = self._progress
                self._result = JobResult(False, error=msg, elapsed_s=time.time() - t0)
            self._set(JobStage.ERROR, progress, msg)

    # ---- the staged body ----
    def _run_pipeline(self, t0):
        import torch
        from PIL import Image

        cfg = self.cfg
        count = num_gaussians_valid(cfg.num_gaussians)

        self._set(JobStage.LOADING_MODEL, 0.05, "Loading model...")
        self._check_cancel()
        pipe = pipeline_loader.get_pipeline()
        gen = torch.Generator(device="cuda").manual_seed(cfg.seed)

        # Reuse the cached latent if its signature still matches (re-decode only).
        sig = latent_signature(cfg, self._preprocessed_id) if self._preprocessed_id else None
        reuse = self._latent is not None and sig is not None and sig == self._signature

        if not reuse:
            self._set(JobStage.PREPROCESS, 0.15, "Preprocessing image...")
            self._check_cancel()
            if self._matte_pil is None:
                src = Image.open(cfg.image_path).convert("RGB")
                self._matte_pil, _ = preprocess.run_preprocess(pipe, src, cfg.erode_radius)
                self._preprocessed_id = f"{cfg.image_path}|{cfg.erode_radius}"

            self._set(JobStage.ENCODE, 0.25, "Encoding image...")
            self._check_cancel()
            cond = pipe.encode_image(self._matte_pil, generator=gen)

            self._set(JobStage.SAMPLE, 0.30, "Sampling latent...")
            self._check_cancel()

            def _cb(step, total):
                if self._is_cancelled():
                    raise _Cancelled()
                frac = 0.30 + 0.45 * (step / max(1, total))
                self._set(JobStage.SAMPLE, frac, f"Sampling {step}/{total}...")

            out = pipe.sample_latent(
                cond, steps=cfg.steps, guidance_scale=cfg.guidance_scale,
                shift=cfg.shift, generator=gen, callback=_cb,
            )
            self._latent = out["latent"]
            self._signature = latent_signature(cfg, self._preprocessed_id)
        else:
            self._log_line("Reusing cached latent (re-decode only).")

        self._set(JobStage.DECODE, 0.80, f"Decoding {count} gaussians...")
        self._check_cancel()
        self._gaussian = pipe.decode_latent(self._latent, num_gaussians=count)

        self._set(JobStage.INSERTING, 0.92, "Inserting into scene...")
        self._check_cancel()
        # Marshal the scene mutation to the UI thread (see __init__ helper).
        node_holder = {}
        done = threading.Event()

        def _ui_insert():
            try:
                node_holder["name"] = insertion.insert_gaussian(
                    self._gaussian, append=cfg.append, log=self._log_line)
            finally:
                done.set()

        lf.ui.schedule_on_ui_thread(_ui_insert)  # verify: stub signature
        done.wait(timeout=30.0)

        with self._lock:
            self._result = JobResult(
                True, elapsed_s=time.time() - t0, num_gaussians=count,
                node_name=node_holder.get("name") or "")
        self._set(JobStage.DONE, 1.0, "Done")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_job_cache.py -v`
Expected: 5 passed. (If `import lichtfeld` fails at collection, gate that import — see Step 5.)

- [ ] **Step 5: If `lichtfeld` import breaks test collection, make it test-safe**

At the top of `core/job.py`, the `import lichtfeld as lf` runs at import time. The pure functions under test (`JobConfig`, `latent_signature`, `num_gaussians_valid`) don't need it. If collection fails outside LFS, wrap it:

```python
try:
    import lichtfeld as lf
except Exception:  # pragma: no cover - outside LFS host
    lf = None  # type: ignore[assignment]
```

Re-run Step 4; expected 5 passed.

- [ ] **Step 6: Commit**

```bash
git add core/job.py tests/test_job_cache.py
git commit -m "feat(core): daemon job with staged pipeline + latent cache + re-decode"
```

---

## Task 8: `__init__.py` (plugin lifecycle)

**Files:**
- Create: `__init__.py`

- [ ] **Step 1: Write `__init__.py`**

Adapted from `Lichtfeld-HYWorld2-Plugin/__init__.py` (same strict ordering). Changes: names → triposplat; drop the Linux gsplat CUDA_HOME block (no gsplat); keep the rest.

```python
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

import lichtfeld as lf  # noqa: E402

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


def on_load():
    downloads.set_logger(lambda msg: lf.log.info(f"[triposplat] {msg}"))
    _apply_perf_flags()
    for cls in _classes:
        lf.register_class(cls)
    downloads.start_background_download()
    try:
        global _last_training_state
        from lfs_plugins.ui.state import AppState
        _last_training_state = bool(AppState.is_training.value)
        AppState.is_training.subscribe_as("triposplat_plugin", _on_training_state_changed)
    except Exception as exc:  # noqa: BLE001
        lf.log.warn(f"triposplat_plugin: couldn't subscribe to is_training ({exc}).")
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
```

- [ ] **Step 2: Ruff check** (cannot import outside LFS, but lint it)

Run: `uvx ruff check __init__.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add __init__.py
git commit -m "feat: plugin lifecycle (HF cache, sys.path, register, training auto-unload)"
```

---

## Task 9: Panel — `panels/main_panel.rcss`

**Files:**
- Create: `panels/main_panel.rcss`

- [ ] **Step 1: Write `panels/main_panel.rcss`**

Copy verbatim from `lichtfeld-vggt-omega/panels/main_panel.rcss` (the full stylesheet — `body`, `#content-wrap`, `.panel-title`, `.section-*`, `.setting-row`, `.prop-label`, `.slider-value`, `.button-row`, `.progress-wrap`, `.job-progress`, `.status-error`, `.log-box`, `.model-banner`, `.log-output`). Then append a preview block:

```css
.preview-wrap {
    margin: 4dp 0 8dp 0;
    display: flex;
    justify-content: center;
}
```

- [ ] **Step 2: Commit**

```bash
git add panels/main_panel.rcss
git commit -m "feat(panel): rcss (vggt-omega base + preview block)"
```

---

## Task 10: Panel — `panels/main_panel.rml`

**Files:**
- Create: `panels/main_panel.rml`

- [ ] **Step 1: Write `panels/main_panel.rml`**

Mirror the vggt-omega RML structure (head/link, `<body data-model="triposplat">`, content-wrap, download banner, sections, run/cancel state machine, log, result, error). Replace multi-view markup with: image input, matte preview (`<div id="im-root">` inside `data-if="has_matte"`), settings (seed/steps/guidance/density + append checkbox), model controls, placement (T/R/S + gizmo mode), result with Save buttons.

```xml
<rml>
<head>
  <link type="text/rcss" href="main_panel.rcss"/>
</head>
<body data-model="triposplat">
<div id="content-wrap">
<div id="content">

  <span class="panel-title text-accent">TripoSplat</span>
  <span class="panel-subtitle text-muted">Generate a 3D Gaussian splat from a single image.</span>

  <!-- Weight download banner -->
  <div class="model-banner" data-if="model_downloading">
    <span class="status-line text-accent">Downloading TripoSplat weights...</span>
    <div class="progress-wrap">
      <progress class="job-progress" max="1" data-attr-value="dl_progress_value"></progress>
      <span class="progress-text">{{dl_progress_pct}} - {{dl_bytes_line}}</span>
    </div>
  </div>
  <div class="model-banner banner-warn" data-if="model_error">
    <span class="status-error">Weight download failed</span>
    <span class="error-detail text-default">{{dl_error_text}}</span>
    <button class="btn btn--warning btn--full" data-event-click="retry_download">Retry</button>
  </div>

  <div class="separator"></div>

  <!-- Input -->
  <span class="section-title text-accent">Input image</span>
  <div class="setting-row">
    <span class="prop-label">Path</span>
    <input type="text" data-value="image_path"/>
  </div>
  <div class="button-row">
    <button class="btn btn--default btn--full" data-event-click="browse_image">Browse...</button>
  </div>

  <!-- Matte preview -->
  <div data-if="has_matte">
    <span class="section-title text-accent">Preprocessed matte</span>
    <div class="preview-wrap"><div id="im-root"></div></div>
    <div class="setting-row">
      <span class="prop-label">Erode radius</span>
      <input type="range" class="setting-slider" data-value="erode_radius" min="0" max="8" step="1"/>
      <span class="slider-value">{{erode_radius}}</span>
    </div>
  </div>

  <!-- Settings -->
  <div class="section-header" data-event-click="toggle_section('settings')">
    <span class="section-arrow" id="arrow-settings">&#x25B6;</span><span>Settings</span>
  </div>
  <div class="section-content collapsed" id="sec-settings">
    <div class="setting-row"><span class="prop-label">Seed</span>
      <input type="text" data-value="seed"/></div>
    <div class="setting-row"><span class="prop-label">Steps</span>
      <input type="range" class="setting-slider" data-value="steps" min="1" max="50" step="1"/>
      <span class="slider-value">{{steps}}</span></div>
    <div class="setting-row"><span class="prop-label">Guidance</span>
      <input type="range" class="setting-slider" data-value="guidance_scale" min="1.0" max="10.0" step="0.5"/>
      <span class="slider-value">{{guidance_scale}}</span></div>
    <div class="setting-row"><span class="prop-label">Gaussians</span>
      <select data-value="num_gaussians">
        <option value="32768">32,768</option>
        <option value="65536">65,536</option>
        <option value="131072">131,072</option>
        <option value="262144">262,144 (max)</option>
      </select></div>
    <div class="setting-row"><span class="prop-label">Shift (adv.)</span>
      <input type="range" class="setting-slider" data-value="shift" min="1.0" max="6.0" step="0.5"/>
      <span class="slider-value">{{shift}}</span></div>
    <div class="setting-row">
      <label class="setting-label"><input type="checkbox" data-checked="append_mode"/><span>Append (keep previous objects)</span></label>
    </div>
    <div class="button-row" data-if="has_latent">
      <button class="btn btn--default btn--full" data-event-click="redecode">Re-decode at this count</button>
    </div>
  </div>

  <!-- Model -->
  <span class="section-title text-accent">Model</span>
  <span class="help-text text-muted">{{model_status_line}}</span>
  <div class="button-row">
    <button class="btn btn--default" data-event-click="reload_weights">Re-download</button>
    <button class="btn btn--default" data-event-click="unload_model" data-if="model_loaded">Unload</button>
  </div>

  <!-- Placement -->
  <div data-if="has_node">
    <span class="section-title text-accent">Placement</span>
    <div class="setting-row"><span class="prop-label">Gizmo</span>
      <select data-value="gizmo_mode">
        <option value="translate">Move</option>
        <option value="rotate">Rotate</option>
        <option value="scale">Scale</option>
      </select></div>
    <div class="setting-row"><span class="prop-label">Pos X/Y/Z</span>
      <input type="text" data-value="tx"/><input type="text" data-value="ty"/><input type="text" data-value="tz"/></div>
    <div class="setting-row"><span class="prop-label">Rot X/Y/Z</span>
      <input type="text" data-value="rx"/><input type="text" data-value="ry"/><input type="text" data-value="rz"/></div>
    <div class="setting-row"><span class="prop-label">Scale</span>
      <input type="text" data-value="scl"/></div>
  </div>

  <!-- Run / Cancel -->
  <div data-if="show_idle">
    <button class="btn btn--success btn--full action-btn" data-event-click="do_start" data-attrif-disabled="!can_run">Generate</button>
  </div>
  <div data-if="show_running">
    <span class="status-line text-default">Stage: {{stage_text}}</span>
    <div class="progress-wrap">
      <progress class="job-progress" max="1" data-attr-value="progress_value"></progress>
      <span class="progress-text">{{progress_pct}}</span>
    </div>
    <span class="help-text text-muted">{{progress_status}}</span>
    <button class="btn btn--warning btn--full action-btn" data-event-click="do_cancel">Cancel</button>
  </div>

  <!-- Log -->
  <div data-if="show_logs">
    <div class="separator"></div>
    <span class="section-title text-accent">Log</span>
    <div class="log-box"><span class="log-output text-muted">{{live_log_text}}</span></div>
  </div>

  <!-- Result -->
  <div data-if="show_results">
    <div class="separator"></div>
    <span class="section-title text-accent">Result</span>
    <span class="result-line text-default">{{result_count}} gaussians in {{result_time}}</span>
    <div class="setting-row"><span class="prop-label">Save format</span>
      <select data-value="save_format"><option value="ply">.ply</option><option value="splat">.splat</option></select></div>
    <div class="button-row">
      <button class="btn btn--default btn--full" data-event-click="save_to_disk">Save to disk</button>
    </div>
  </div>

  <!-- Error -->
  <div data-if="show_error">
    <div class="separator"></div>
    <span class="status-error">Error</span>
    <span class="error-detail text-default">{{error_text}}</span>
  </div>

</div>
</div>
</body>
</rml>
```

- [ ] **Step 2: Commit**

```bash
git add panels/main_panel.rml
git commit -m "feat(panel): rml layout (input, matte, settings, placement, run, result)"
```

---

## Task 11: Panel — `panels/main_panel.py`

**Files:**
- Create: `panels/main_panel.py`

This is the largest adapted file. Build it incrementally; commit when it lints clean. Mirror the vggt-omega panel's structure (dirty groups, `on_bind_model`, diff-and-dirty `on_update`, run/cancel state machine, daemon model-load thread). Add: the matte `draw(ui)`, the gizmo two-way binding, the latent-cache hand-off between jobs.

- [ ] **Step 1: Write the module constants + class skeleton + lifecycle**

```python
from __future__ import annotations

import threading
from pathlib import Path

import lichtfeld as lf

from core import downloads, export, pipeline_loader, preprocess
from core.job import JobConfig, JobResult, TripoSplatJob, latent_signature, num_gaussians_valid

PLUGIN_NAME = "triposplat_plugin"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_DIRTY_DL = ("model_downloading", "model_error", "dl_progress_value", "dl_progress_pct",
             "dl_bytes_line", "dl_error_text", "model_status_line", "model_loaded", "can_run")
_DIRTY_PREVIEW = ("has_matte",)
_DIRTY_RUN = ("stage_text", "progress_value", "progress_pct", "progress_status")
_DIRTY_RUNNING = ("show_idle", "show_running", "can_run")
_DIRTY_LOG = ("show_logs", "live_log_text")
_DIRTY_RESULT = ("show_results", "show_error", "error_text", "result_count",
                 "result_time", "has_node", "has_latent")
_DIRTY_PLACE = ("tx", "ty", "tz", "rx", "ry", "rz", "scl", "gizmo_mode")


class TripoSplatPanel(lf.ui.Panel):
    id = "triposplat.main"
    label = "TripoSplat"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 226
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 150

    def __init__(self):
        self._doc = None
        self._handle = None
        # inputs / settings
        self.image_path = ""
        self.seed = 42
        self.steps = 20
        self.guidance_scale = 3.0
        self.shift = 3.0
        self.num_gaussians = 262144
        self.erode_radius = 1
        self.append_mode = False
        self.save_format = "ply"
        # placement
        self.tx = self.ty = self.tz = 0.0
        self.rx = self.ry = self.rz = 0.0
        self.scl = 1.0
        self.gizmo_mode = "translate"
        # runtime
        self._job: TripoSplatJob | None = None
        self._last_result: JobResult | None = None
        self._matte_pil = None
        self._matte_rgb = None           # np.uint8 [H,W,3] for ui.image_tensor
        self._preprocessed_id = ""
        self._cached_latent = None
        self._cached_signature = None
        self._node_name = ""
        self._gizmo = None
        self._collapsed = {"settings"}
        # diff caches
        self._last_dl = None
        self._last_stage = ""
        self._last_progress = -1.0
        self._last_status = ""
        self._last_log = ""
        self._last_running = False
        self._last_result_key = None

    def _dirty(self, *fields):
        if not self._handle:
            return
        for name in fields:
            self._handle.dirty(name)

    # ---- lifecycle ----
    def on_mount(self, doc):
        self._doc = doc
        self._sync_section_states()

    def on_unmount(self, doc):
        if self._job and self._job.is_running():
            self._job.cancel()
        self._detach_gizmo()
        try:
            lf.ui.free_plugin_textures(PLUGIN_NAME)  # verify: stub
        except Exception:
            pass
        doc.remove_data_model("triposplat")
        self._doc = None
        self._handle = None
```

- [ ] **Step 2: Add `on_bind_model` (all bindings)**

```python
    def on_bind_model(self, ctx):
        model = ctx.create_data_model("triposplat")
        if model is None:
            return
        # two-way scalar bindings
        model.bind("image_path", lambda: self.image_path, self._set_image_path)
        model.bind("seed", lambda: str(self.seed), lambda v: self._set_int("seed", v, 0, 2**31 - 1))
        model.bind("steps", lambda: str(self.steps), lambda v: self._set_int("steps", v, 1, 50))
        model.bind("guidance_scale", lambda: f"{self.guidance_scale:.1f}",
                   lambda v: self._set_float("guidance_scale", v, 1.0, 10.0))
        model.bind("shift", lambda: f"{self.shift:.1f}", lambda v: self._set_float("shift", v, 1.0, 6.0))
        model.bind("num_gaussians", lambda: str(self.num_gaussians),
                   lambda v: self._set_int("num_gaussians", v, 32768, 262144))
        model.bind("erode_radius", lambda: str(self.erode_radius), self._set_erode)
        model.bind("append_mode", lambda: self.append_mode, lambda v: self._set_bool("append_mode", v))
        model.bind("save_format", lambda: self.save_format, lambda v: setattr(self, "save_format", str(v)))
        model.bind("gizmo_mode", lambda: self.gizmo_mode, self._set_gizmo_mode)
        for ax in ("tx", "ty", "tz", "rx", "ry", "rz", "scl"):
            model.bind(ax, (lambda a=ax: f"{getattr(self, a):.3f}"),
                       (lambda v, a=ax: self._set_transform_field(a, v)))
        # computed one-way
        model.bind_func("model_downloading", lambda: downloads.get_state()["stage"] == "downloading")
        model.bind_func("model_error", lambda: downloads.get_state()["stage"] == "error")
        model.bind_func("dl_progress_value", lambda: downloads.get_state()["progress"])
        model.bind_func("dl_progress_pct", lambda: f"{int(downloads.get_state()['progress']*100)}%")
        model.bind_func("dl_bytes_line", self._dl_bytes_line)
        model.bind_func("dl_error_text", lambda: downloads.get_state()["error"])
        model.bind_func("model_status_line", self._model_status_line)
        model.bind_func("model_loaded", pipeline_loader.is_loaded)
        model.bind_func("has_matte", lambda: self._matte_rgb is not None)
        model.bind_func("has_latent", lambda: self._cached_latent is not None)
        model.bind_func("has_node", lambda: bool(self._node_name))
        model.bind_func("can_run", self._can_run)
        model.bind_func("show_idle", lambda: not self._is_running())
        model.bind_func("show_running", self._is_running)
        model.bind_func("stage_text", lambda: self._job.stage.value if self._job else "")
        model.bind_func("progress_value", lambda: self._job.progress if self._job else 0.0)
        model.bind_func("progress_pct", lambda: f"{int((self._job.progress if self._job else 0)*100)}%")
        model.bind_func("progress_status", lambda: self._job.status if self._job else "")
        model.bind_func("show_logs", lambda: bool(self._job and self._job.log_text))
        model.bind_func("live_log_text", lambda: self._job.log_text if self._job else "")
        model.bind_func("show_results", self._show_results)
        model.bind_func("show_error", self._show_error)
        model.bind_func("error_text", lambda: self._last_result.error if self._last_result else "")
        model.bind_func("result_count", lambda: str(self._last_result.num_gaussians) if self._last_result else "")
        model.bind_func("result_time", lambda: f"{self._last_result.elapsed_s:.1f}s" if self._last_result else "")
        # events
        model.bind_event("browse_image", self._on_browse_image)
        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_event("retry_download", lambda *_: downloads.start_background_download())
        model.bind_event("reload_weights", self._on_reload_weights)
        model.bind_event("unload_model", lambda *_: threading.Thread(target=pipeline_loader.unload, daemon=True).start())
        model.bind_event("do_start", self._on_start)
        model.bind_event("do_cancel", self._on_cancel)
        model.bind_event("redecode", self._on_redecode)
        model.bind_event("save_to_disk", self._on_save)
        self._handle = model.get_handle()
```

- [ ] **Step 3: Add `draw(ui)` for the matte + `on_update` diff-and-dirty**

```python
    def draw(self, ui):
        # Rendered into <div id="im-root">. Show the matte if present.
        if self._matte_rgb is None:
            return
        h, w = self._matte_rgb.shape[0], self._matte_rgb.shape[1]
        avail_w, _ = ui.get_content_region_avail()
        disp_w = max(64.0, min(float(avail_w), 320.0))
        disp_h = disp_w * (h / max(1, w))
        # verify: ui.image_tensor(label, array_or_tensor, (w,h)) — confirm signature in stubs.
        ui.image_tensor("triposplat_matte", self._matte_rgb, (disp_w, disp_h))

    def on_update(self, doc):
        del doc
        dirty = False
        dl = downloads.get_state()
        dl_key = (dl["stage"], round(dl["progress"], 3), pipeline_loader.is_loaded())
        if dl_key != self._last_dl:
            self._last_dl = dl_key
            self._dirty(*_DIRTY_DL)
            dirty = True
        job = self._job
        if job:
            triple = (job.stage.value, job.progress, job.status)
            if triple != (self._last_stage, self._last_progress, self._last_status):
                self._last_stage, self._last_progress, self._last_status = triple
                self._dirty(*_DIRTY_RUN)
                dirty = True
            if job.log_text != self._last_log:
                self._last_log = job.log_text
                self._dirty(*_DIRTY_LOG)
                dirty = True
            if job.is_running() != self._last_running:
                self._last_running = job.is_running()
                self._dirty(*_DIRTY_RUNNING)
                dirty = True
            rk = self._result_key(job.result)
            if rk is not None and rk != self._last_result_key:
                self._last_result = job.result
                self._last_result_key = rk
                self._on_job_finished(job)
                self._dirty(*_DIRTY_RESULT, *_DIRTY_PREVIEW, *_DIRTY_PLACE)
                dirty = True
        return dirty

    @staticmethod
    def _result_key(r):
        if r is None:
            return None
        return (r.success, r.num_gaussians, round(r.elapsed_s, 2), r.node_name, r.error)
```

- [ ] **Step 4: Add handlers, setters, gizmo, and job hand-off**

```python
    def _is_running(self):
        return self._job is not None and self._job.is_running()

    def _can_run(self):
        return bool(self.image_path) and downloads.is_ready() and not self._is_running()

    def _show_results(self):
        return self._last_result is not None and self._last_result.success

    def _show_error(self):
        return self._last_result is not None and not self._last_result.success and self._last_result.error != "Cancelled"

    def _dl_bytes_line(self):
        s = downloads.get_state()
        return f"{s['bytes_downloaded']//1_000_000} / {s['bytes_total']//1_000_000} MB"

    def _model_status_line(self):
        if downloads.is_ready():
            return "Model ready (loaded)" if pipeline_loader.is_loaded() else "Weights ready (not loaded)"
        return downloads.get_state().get("message", "")

    # --- input ---
    def _set_image_path(self, v):
        self.image_path = str(v)
        self._kick_preprocess()

    def _on_browse_image(self, *_):
        path = lf.ui.open_image_dialog("")  # verify: stub returns '' on cancel
        if path:
            self.image_path = path
            self._dirty("image_path")
            self._kick_preprocess()

    def _set_erode(self, v):
        self._set_int("erode_radius", v, 0, 8)
        self._kick_preprocess()

    def _kick_preprocess(self):
        """Run preprocess on a daemon thread; result feeds the matte preview."""
        if not self.image_path or not downloads.is_ready():
            return
        path, erode = self.image_path, self.erode_radius

        def _task():
            try:
                from PIL import Image
                pipe = pipeline_loader.get_pipeline()
                src = Image.open(path).convert("RGB")
                matte_pil, matte_rgb = preprocess.run_preprocess(pipe, src, erode)
                self._matte_pil, self._matte_rgb = matte_pil, matte_rgb
                self._preprocessed_id = f"{path}|{erode}"
                # changing the matte invalidates any cached latent
                self._cached_latent = None
                self._cached_signature = None
                self._dirty(*_DIRTY_PREVIEW)
            except Exception as exc:  # noqa: BLE001
                lf.log.warn(f"[triposplat] preprocess failed: {exc}")

        threading.Thread(target=_task, daemon=True).start()

    # --- run / cancel / redecode ---
    def _new_job(self, *, reuse_latent):
        cfg = JobConfig(image_path=self.image_path, seed=self.seed, steps=self.steps,
                        guidance_scale=self.guidance_scale, shift=self.shift,
                        num_gaussians=num_gaussians_valid(self.num_gaussians),
                        erode_radius=self.erode_radius, append=self.append_mode)
        job = TripoSplatJob(
            cfg,
            cached_latent=self._cached_latent if reuse_latent else None,
            cached_signature=self._cached_signature if reuse_latent else None,
            matte_pil=self._matte_pil,
            preprocessed_id=self._preprocessed_id,
        )
        self._last_result = None
        self._last_result_key = None
        self._last_log = ""
        self._job = job
        job.start()
        self._dirty(*_DIRTY_RUNNING, *_DIRTY_LOG, *_DIRTY_RESULT, *_DIRTY_RUN)

    def _on_start(self, *_):
        if not self._can_run():
            lf.log.warn("[triposplat] Cannot run: need an image and downloaded weights.")
            return
        self._new_job(reuse_latent=False)

    def _on_redecode(self, *_):
        if self._cached_latent is None or self._is_running():
            return
        self._new_job(reuse_latent=True)

    def _on_cancel(self, *_):
        if self._job and self._job.is_running():
            self._job.cancel()

    def _on_job_finished(self, job):
        if job.result and job.result.success:
            self._cached_latent = job.latent
            self._cached_signature = job.signature
            self._node_name = job.result.node_name
            self._reset_placement_fields()
            self._attach_gizmo()

    # --- save ---
    def _on_save(self, *_):
        if not (self._job and self._job.gaussian):
            return
        path = lf.ui.save_file_dialog("", f"triposplat.{self.save_format}")  # verify: stub
        if path:
            try:
                export.save(self._job.gaussian, path, self.save_format)
                lf.log.info(f"[triposplat] saved {path}")
            except Exception as exc:  # noqa: BLE001
                lf.log.error(f"[triposplat] save failed: {exc}")

    def _on_reload_weights(self, *_):
        downloads.delete_models()
        downloads.start_background_download()

    # --- placement / gizmo ---
    def _attach_gizmo(self):
        if not self._node_name:
            return
        self._detach_gizmo()
        try:
            self._gizmo = lf.TransformGizmo()  # verify: stub ctor + attach/operation API
            self._gizmo.operation = self.gizmo_mode
            self._gizmo.attach_to_node(self._node_name, visualizer_world=False)
            self._gizmo.set_on_change(self._on_gizmo_change)
            self._gizmo.set_on_end(self._on_gizmo_end)
        except Exception as exc:  # noqa: BLE001
            lf.log.warn(f"[triposplat] gizmo attach failed: {exc}")

    def _detach_gizmo(self):
        if self._gizmo is not None:
            try:
                self._gizmo.detach()
            except Exception:
                pass
            self._gizmo = None

    def _on_gizmo_change(self, *_):
        try:
            d = lf.decompose_transform(self._gizmo.matrix)
            self.tx, self.ty, self.tz = d["translation"]
            self.rx, self.ry, self.rz = d["euler"]
            self.scl = d["scale"][0]
            self._dirty(*_DIRTY_PLACE)
        except Exception:
            pass

    def _on_gizmo_end(self, *_):
        self._on_gizmo_change()

    def _set_gizmo_mode(self, v):
        self.gizmo_mode = str(v)
        if self._gizmo is not None:
            try:
                self._gizmo.operation = self.gizmo_mode
            except Exception:
                pass

    def _set_transform_field(self, axis, v):
        try:
            setattr(self, axis, float(v))
        except (TypeError, ValueError):
            return
        if not self._node_name:
            return
        m = lf.compose_transform((self.tx, self.ty, self.tz),
                                 (self.rx, self.ry, self.rz),
                                 (self.scl, self.scl, self.scl))
        lf.set_node_transform(self._node_name, m)
        lf.ui.request_redraw()

    def _reset_placement_fields(self):
        self.tx = self.ty = self.tz = 0.0
        self.rx = self.ry = self.rz = 0.0
        self.scl = 1.0

    # --- generic setters / sections ---
    def _set_int(self, name, v, lo, hi):
        try:
            setattr(self, name, max(lo, min(hi, int(float(v)))))
        except (TypeError, ValueError):
            pass

    def _set_float(self, name, v, lo, hi):
        try:
            setattr(self, name, max(lo, min(hi, float(v))))
        except (TypeError, ValueError):
            pass

    def _set_bool(self, name, v):
        setattr(self, name, bool(v))

    def _on_toggle_section(self, handle, event, args):
        del handle, event
        name = args[0] if args else ""
        if name in self._collapsed:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)
        self._sync_section_states()

    def _sync_section_states(self):
        if not self._doc:
            return
        for name in ("settings",):
            content = self._doc.get_element_by_id(f"sec-{name}")
            arrow = self._doc.get_element_by_id(f"arrow-{name}")
            if content:
                content.set_class("collapsed", name in self._collapsed)
            if arrow:
                arrow.set_class("is-expanded", name not in self._collapsed)
```

- [ ] **Step 5: Ruff check**

Run: `uvx ruff check panels/main_panel.py`
Expected: no errors (fix imports/lint as needed).

- [ ] **Step 6: Commit**

```bash
git add panels/main_panel.py
git commit -m "feat(panel): TripoSplat hybrid panel (matte preview, settings, placement, run, save)"
```

---

## Task 12: Full test run + README + manual smoke

**Files:**
- Create: `README.md`
- Modify: none

- [ ] **Step 1: Run the whole GPU-free suite**

Run: `uv run pytest -v`
Expected: all tests in `test_downloads.py`, `test_insertion.py`, `test_job_cache.py` pass.

- [ ] **Step 2: Ruff the whole project**

Run: `uvx ruff check .`
Expected: no errors (the vendored `triposplat/` is excluded by `[tool.ruff].extend-exclude`).

- [ ] **Step 3: Write `README.md`**

Mirror the sibling README section order: Title + one-liner; Requirements (NVIDIA GPU, compute ≥ 8.0, ~3.8 GB weights auto-download, ~4 GB venv); Install (`install.ps1` / `install.sh` → launch LFS → first-run `uv sync`); Usage (Browse image → matte preview → settings → Generate → place with gizmo/fields → optional Save); Settings reference (seed/steps/guidance/gaussians/erode/shift, append mode, re-decode); Notes (MIT code + MIT weights, weights not redistributed); Uninstall.

- [ ] **Step 4: Manual smoke checklist** (in LichtFeld Studio, with GPU)

```
[ ] install.ps1, launch LFS, wait for uv sync; log shows "triposplat_plugin loaded"
[ ] Weight download banner progresses to ready (~3.8 GB)
[ ] Browse a product photo -> matte preview renders (RGB on black)
[ ] Adjust erode radius -> matte re-renders
[ ] Generate -> staged progress (preprocess->encode->sample->decode->inserting), splat appears at origin in current scene
[ ] Generate into a scene that already has content -> object adds without clearing existing nodes
[ ] Change Gaussians to 65,536 -> "Re-decode at this count" is near-instant (no re-sample), node replaced
[ ] Move/Rotate/Scale via gizmo -> T/R/S fields update; edit a field -> gizmo + node move
[ ] Enable Append, Generate again -> TripoSplat_01 created alongside the first
[ ] Save to disk (.ply and .splat) -> files open correctly in a viewer
[ ] Start LFS training -> model auto-unloads (VRAM frees); log notes it
[ ] Disable plugin -> clean unload, no crash
```

- [ ] **Step 5: Add an opt-in GPU integration test `tests/test_gpu_smoke.py`**

```python
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA", allow_module_level=True)
pytest.importorskip("triposplat")


@pytest.mark.gpu
def test_end_to_end_decode_produces_finite_gaussian():
    import sys
    sys.path.insert(0, ".")
    from PIL import Image
    from core import downloads, pipeline_loader
    if not downloads.is_weights_cached():
        pytest.skip("weights not downloaded")
    pipe = pipeline_loader.get_pipeline()
    img = Image.new("RGB", (512, 512), (180, 180, 180))
    g, _prepared = pipe.run(img, seed=0, steps=4, num_gaussians=32768)
    xyz = g.get_xyz
    assert xyz.shape[0] > 0
    assert torch.isfinite(xyz).all()
    assert torch.isfinite(g.get_opacity).all()
```

- [ ] **Step 6: Commit**

```bash
git add README.md tests/test_gpu_smoke.py
git commit -m "docs: README + opt-in GPU smoke test"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** Inference local-only (Task 4 ctor, bf16 guard) ✓; full pipeline UX — matte preview (Tasks 5,10,11 draw/ui.image_tensor), re-decode/Learned-Density (Task 7 `decode_only` via reuse_latent + Task 11 redecode), save-to-disk (Tasks 5,11) ✓; placement origin + T/R/S + gizmo (Task 11) ✓; import-and-view only — no training handoff ✓; replace-default + append toggle (Task 6 `next_group_name`, Task 11 `append_mode`) ✓; auto background weight download (Tasks 3,8) ✓; mirror structure/naming (`order=226`, `triposplat_plugin` link, data-model `triposplat`) ✓; threading marshaled insert (Task 7 `schedule_on_ui_thread`) ✓; teardown (Task 8) ✓; tests (Tasks 3,6,7,12) ✓.

**Placeholder scan:** No TBD/TODO. `# verify:` markers denote host-API contracts to confirm against stubs/ml-sharp (external APIs, not gaps in our code): `lf.io.load().splat_data.*_raw`, `Scene.has_node`, `lf.ui.schedule_on_ui_thread`, `lf.ui.open_image_dialog`/`save_file_dialog`, `lf.ui.image_tensor`/`free_plugin_textures`, `lf.TransformGizmo` + `compose/decompose/set_node_transform`. All have a stated fallback.

**Type consistency:** `JobConfig`/`JobResult`/`TripoSplatJob`/`latent_signature`/`num_gaussians_valid` names match across Tasks 6,7,11. `downloads.ckpt_paths()` keys match the `pipeline_loader` ctor kwargs. `insertion.insert_gaussian(gaussian, *, append, log)` / `next_group_name(scene, append)` match Tasks 6,7. Panel dirty-group tuples are defined once and reused.

**Open verification items carried to implementation:** the `# verify:` host APIs above, and Task 2's confirmation of the real upstream method/accessor names against the pinned commit.
