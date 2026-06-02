# TripoSplat — LichtFeld Studio Plugin (Design Spec)

**Date:** 2026-06-01 · **Author:** Yehe Liu · **Status:** Approved for planning

Single image → preprocess (matte) → settings → generate a 3D Gaussian splat → insert into the (possibly existing) LFS scene, with interactive placement and optional `.ply`/`.splat` export. This plugin is a faithful **mirror of the HYWorld2 and VGGT-Omega plugins** in structure, conventions, and load-bearing scene-insertion details, adapting only what single-image generation actually requires.

---

## 1. Overview

TripoSplat wraps the upstream **VAST-AI-Research/TripoSplat** single-image generative pipeline (DINOv3 ViT-H + Flux2 VAE encode → flow-matching denoiser with CFG → Octree Gaussian decoder) and pushes the decoded Gaussians **directly into the live LFS scene** zero-copy via DLPack, with the same activation-inversion / SH-layout / wxyz-quaternion contract the two sibling insertion plugins ship. It adds three things the multi-view siblings never needed:

1. **A preprocessed-matte preview** (the upstream `preprocess_image` output is a 1024² RGBA tensor), rendered in-panel.
2. **Learned-Density re-decode** — a sampled latent is cached so the decoder can be replayed at a different `num_gaussians` near-instantly without re-sampling.
3. **Interactive placement** — origin insertion plus T/R/S fields and a native `lf.TransformGizmo`, wired as **one bound editor** of the splat node's transform.

After insertion the plugin is **import-and-view only** — no `prepare_training_from_scene`, no `set_training_model_node`, no COLMAP/dataset writer.

Target hardware: **RTX 4090, 24 GB, compute 8.9, cu130**. VRAM is a non-issue at the max 262 144 gaussians, so the siblings' per-frame VRAM calibration module is dropped.

---

## 2. Locked decisions

| Decision | Resolution |
|---|---|
| **Inference** | Local GPU only — in-process torch, no cloud / HF-Space / `gradio_client` fallback. CUDA-only (upstream has no CPU path). |
| **Pipeline UX** | Full: preprocessed matte preview before generate; Learned-Density re-decode (replay decoder from cached latent); save `.ply`/`.splat` to disk. |
| **Placement** | Insert at world origin + T/R/S fields in panel + interactive `lf.TransformGizmo`, as one two-way-bound editor. |
| **After insert** | Import & view only — no training/refine handoff. |
| **GPU / build** | RTX 4090 24 GB, compute 8.9 (bf16-capable); torch/torchvision pinned to **cu130**. |
| **Re-run mode** | **Default = replace-by-name**; an **append toggle** (`TripoSplat_01/02/…`) lets the user accumulate multiple generated objects in one scene. |
| **Weight fetch** | **Auto background download on plugin load** (HYWorld2 behavior), with manual re-download / cancel controls. |
| **Default placement value** | **World origin + identity** transform (upstream `_DEFAULT_TRANSFORM` axis remap only); no auto-fit heuristic. |

---

## 3. Mirror mapping

For each sibling artifact, the corresponding TripoSplat artifact and its provenance. **V** = copied near-verbatim (swap names/IDs only), **A** = adapted, **N** = new (legitimate single-image addition).

| Sibling (HYWorld2 / VGGT-Omega) | TripoSplat | Mode | Notes |
|---|---|---|---|
| `__init__.py` | `__init__.py` | **V** | Copy load/teardown skeleton verbatim; swap names + HF cache subpath; keep `from lfs_plugins.ui.state import AppState` subscribe in try/except. Drop skyseg/onnx/perf-flags specific to multi-view. |
| `pyproject.toml` | `pyproject.toml` | **A** | Same `[project]`/`[tool.lichtfeld]`/cu130 index/ruff shape; TripoSplat dep set (no gsplat/onnxruntime/opencv/imageio-ffmpeg). |
| `conftest.py` (VGGT) | `conftest.py` | **V** | `Package.setup` monkeypatch — copy as-is. |
| `core/__init__.py` | `core/__init__.py` | **V** | Empty marker. |
| `core/downloads.py` (HYWorld2) | `core/downloads.py` | **A** | HYWorld2 is the downloader template. Re-point to `VAST-AI/TripoSplat` (5 files, ungated). Drop skyseg urlretrieve. |
| `core/checkpoints.py` (VGGT) | — | dropped | VGGT's gated-repo handling not needed (TripoSplat is ungated). |
| `core/pipeline_loader.py` | `core/pipeline_loader.py` | **A** | Cached-singleton + unload teardown verbatim; swap `from_pretrained` for the 5-path `TripoSplatPipeline(...)` ctor; add bf16/compute-cap guard. Drop warmup/compile-block/skyseg. |
| `core/job.py` | `core/job.py` | **A** | Threading skeleton (daemon, lock-guarded props, deque log, `_check_cancel`, `JobConfig`/`JobResult`, exception→failed-result) verbatim; new staged body with **latent cache** + `decode_only` entry. |
| `core/direct_output.py` | `core/direct_output.py` | **A** | The verbatim scene-insertion half: `_to_lf` (DLPack + numpy fallback), `remove_node`→`add_group`→`add_splat`, activation inversion, SH layout, wxyz quats. Drop `_add_cameras`/`_add_point_cloud`/`add_splats_from_points`/`set_training_node`. Add `_DEFAULT_TRANSFORM` axis-remap applied by us. |
| `core/colmap_io.py` | — | dropped | No multi-camera dataset. |
| `core/vram_profile.py` / `core/vram.py` | — | dropped | Single fixed-resolution generation; VRAM is a non-issue. |
| `core/frames.py` (VGGT) | — | dropped | Single image; no folder/video. |
| `core/inference.py` (VGGT) | — | dropped | Replaced by the generative pipeline inside `pipeline_loader`/`job`. |
| — | `core/preprocess.py` | **N** | Thin wrapper around `pipeline.preprocess_image`, returns the 1024² RGBA matte tensor + erode param. |
| — | `core/export.py` | **N** | `save_ply`/`save_splat` to disk (siblings have no export). |
| `panels/main_panel.py` | `panels/main_panel.py` | **A** | Hybrid-RML scaffold verbatim (on_bind_model / on_update diff-and-dirty / on_unmount / dirty groups / run-cancel state machine); new fields (seed/steps/guidance/density/placement/append), `draw(self, ui)` for the matte, gizmo two-way sync. |
| `panels/main_panel.rml` | `panels/main_panel.rml` | **A** | Same section vocabulary; add `<div id="im-root">` preview region, T/R/S rows, density dropdown, re-decode button, append checkbox. Drop all multi-view/subset markup. |
| `panels/main_panel.rcss` | `panels/main_panel.rcss` | **V** | Generic dark hybrid stylesheet — copy as-is. |
| vendored `hyworld2/` / `thrid_party/vggt-omega/` | `triposplat/` | **A** | Vendor upstream `triposplat.py` + `model.py` at a pinned commit; on `sys.path`; ruff-excluded. |
| `install.ps1` / `.sh`, `uninstall.ps1` / `.sh` | same | **V** | HYWorld2-grade defensive scripts (the better template); swap link name + vendored-file preflight path. |
| `README.md` | `README.md` | **A** | Same section order; single-image usage; MIT/MIT note. |
| `LICENSE` (MIT) | `LICENSE` | **V** | MIT, (c) 2026 Yehe Liu. |
| `NOTICE` | `NOTICE` | **A** | Enumerate vendored MIT code + 5 ungated MIT weight files + dep licenses; "weights not redistributed". |
| `.github/workflows/ci.yml` | `.github/workflows/ci.yml` | **A** | structure-integrity + ruff + `uv lock --no-cache` on ubuntu+windows. |
| `.gitignore` | `.gitignore` | **V** | Root-anchored: `/.venv/`, `/models/`, `/cache/`, `/uv.lock`, `__pycache__/`. |
| `tests/` (VGGT) | `tests/` | **A** | GPU-free unit tests (§14). |

---

## 4. File tree

```
lichtfeld-triposplat-plugin/
├── __init__.py                 # entry/lifecycle: stderr guard → HF cache env → sys.path → register → bg download → is_training subscribe → teardown
├── pyproject.toml              # [project] + [tool.lichtfeld] + cu130 index/sources + ruff + pytest
├── conftest.py                 # Package.setup monkeypatch (copied from VGGT verbatim)
├── README.md
├── LICENSE                     # MIT, (c) 2026 Yehe Liu
├── NOTICE                      # MIT vendored code + 5 MIT ungated weights + dep licenses
├── .gitignore                  # root-anchored
├── install.ps1 / install.sh
├── uninstall.ps1 / uninstall.sh
├── .github/workflows/ci.yml
├── triposplat/                 # VENDORED upstream, pinned commit, ruff-excluded, on sys.path
│   ├── __init__.py
│   ├── triposplat.py           # TripoSplatPipeline, Gaussian, _DEFAULT_TRANSFORM, save_ply/save_splat
│   └── model.py                # OctreeGaussianDecoder etc.
├── core/
│   ├── __init__.py
│   ├── downloads.py            # snapshot_download(VAST-AI/TripoSplat) → models/ ; byte-watcher progress ; 5-file completeness
│   ├── pipeline_loader.py      # cached GPU singleton ; mixed dtype ; bf16/compute-cap guard ; load/unload teardown
│   ├── preprocess.py           # preprocess_image → 1024² RGBA matte tensor (NEW)
│   ├── job.py                  # JobConfig/JobResult + TripoSplatJob daemon ; full run + decode_only ; latent cache
│   ├── direct_output.py        # _to_lf (DLPack) + activation inversion + axis remap + scene.add_splat (verbatim half)
│   └── export.py               # save .ply / .splat (NEW)
├── panels/
│   ├── __init__.py
│   ├── main_panel.py           # hybrid RML scaffold + draw(ui) matte + gizmo two-way sync
│   ├── main_panel.rml          # data-bound layout + <div id="im-root"> preview region
│   └── main_panel.rcss
├── tests/
│   ├── test_direct_output_math.py
│   ├── test_transform.py
│   ├── test_job_cache.py
│   └── test_downloads.py
└── models/                     # gitignored weight cache (HF hub under here)
```

**Snake_case naming triplet (pinned):** install junction `triposplat_plugin`; data-model `triposplat`; panel id `triposplat.main`; panel `order = 226` (next to HYWorld2 220 / VGGT 225). `[project].name = "TripoSplat"` (marketplace display).

---

## 5. `pyproject.toml`

```toml
[project]
name = "TripoSplat"
version = "0.1.0"
description = "Generate a 3D Gaussian splat from a single image using VAST-AI TripoSplat, inserted directly into the scene."
requires-python = ">=3.12,<3.13"          # LFS ships vcpkg Python 3.12
authors = [{ name = "Yehe Liu" }]

dependencies = [
    # `triposplat` is vendored inline at ./triposplat/ and put on sys.path by __init__.py.
    "torch==2.11.0",                                       # core (cu130)
    "torchvision==0.26.0",                                 # image transforms / preprocess
    "numpy",
    "safetensors",                                         # weight loading
    "pillow",                                              # image I/O
    "tqdm",                                                # progress (used by upstream sampler)
    "huggingface-hub",                                     # weight download
    "triton-windows>=3.5 ; platform_system == 'Windows'",  # required by torch-cu130 on Windows
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
hot_reload = false          # multi-GB venv + ~3.8 GB weights; restart LFS to pick up edits
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.5.0"
required_features = []
auto_start = false
```

> Exact torch/torchvision pins (`2.11.0` / `0.26.0`) and the cu130 index will be re-confirmed against the LFS host's shipped torch at implementation time; the rule is "match the host's CUDA major for DLPack interop." No `gsplat`, `omegaconf`, `opencv-python`, `onnxruntime*`, `imageio-ffmpeg`, `plyfile`, `einops`, or `gradio_client`.

---

## 6. `__init__.py` — load / teardown order

Import-time (before `on_load`), mirroring the siblings:

1. **stderr EINVAL guard** — flush test; on `OSError` repoint `sys.stderr` to devnull. Defuses tqdm/HF progress crashes under LFS embedded Python on Windows.
2. `_PLUGIN_DIR = Path(__file__).parent`.
3. **HF cache scoping — full isolation** (ungated weights, no token to preserve): set `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE` → `<plugin>/models/huggingface` **before** any `huggingface_hub` import. (Opposite of VGGT, which omits `HF_HOME` to keep a gated token — not our case.)
4. `TORCHINDUCTOR_CACHE_DIR` + `TRITON_CACHE_DIR` → `<plugin>/cache`.
5. `sys.path.insert(0, str(_PLUGIN_DIR))` to expose vendored `triposplat`.
6. `import lichtfeld as lf`; import core + panel; `_classes = [TripoSplatPanel]`.

`on_load()`: set download logger; `_apply_perf_flags()` (cudnn.benchmark, TF32, matmul precision, guarded); `register_class` each; **start auto background weight download** (`downloads.start_background_download()`, idempotent — no-op if all 5 files already present); subscribe to training:
```python
try:
    from lfs_plugins.ui.state import AppState
    _last_training_state = bool(AppState.is_training.value)
    AppState.is_training.subscribe_as("triposplat_plugin", _on_training_state_changed)
except Exception as exc:
    lf.log.warn(f"triposplat_plugin: couldn't subscribe to is_training ({exc}).")
```
Rising edge → unload pipeline on a daemon thread to free VRAM for the trainer.

`on_unload()`: stop training (poll ≤2 s) → cancel download + join → `pipeline_loader.unload()` → `gc.collect()`×2 → `cuda.synchronize()`+`empty_cache()` (guarded) → `unregister_class` reversed.

> **Verified correction:** `AppState` is **not** an `lf` symbol — it is `lfs_plugins.ui.state.AppState`, a reactive Signal (`.value` / `.subscribe_as`), exactly as both shipped siblings import it. We mirror that. `lf.ui.context().is_training` / `lf.get_scene_generation()` are the stub-typed inline reads used in §8.

---

## 7. Core modules

### 7.1 `core/downloads.py` (adapted from HYWorld2)
Thread-safe state + lock, `start_background_download()` / `cancel_download()` / `join()` / `delete_models()`, disk-watcher progress thread. **Auto-started from `on_load`** (idempotent). Single ungated repo `VAST-AI/TripoSplat` (~3.78 GB, 5 files):
- `diffusion_models/triposplat_fp16.safetensors`
- `vae/triposplat_vae_decoder_fp16.safetensors`
- `clip_vision/dino_v3_vit_h.safetensors`
- `vae/flux2-vae.safetensors`
- `background_removal/birefnet.safetensors`

`is_ready()` = all 5 present. No gated handling, no skyseg. Panel shows a download banner (progress + cancel + retry); a manual "Re-download" control is available.

### 7.2 `core/pipeline_loader.py` (adapted)
Module-level `_pipeline`/`_loaded` under `_lock`; double-checked locking. `_assert_cuda_and_bf16()` then construct `TripoSplatPipeline(ckpt_path=..., decoder_path=..., dinov3_path=..., flux2_vae_encoder_path=..., rmbg_path=..., device="cuda")` from explicit local paths (no `from_pretrained`, no CPU path). `unload()`: drop ref, gc×2, sync+empty_cache. No warmup/compile-block.

### 7.3 `core/preprocess.py` (new)
`run_preprocess(pipeline, pil_image, erode_radius=1) -> matte_rgba` → PIL 1024² RGBA → contiguous `[H,W,4]` tensor for preview. Worker-side; re-run only on image/erode change.

### 7.4 `core/job.py` — daemon job with latent cache
`JobStage`: IDLE/PREPARING/LOADING_MODEL/PREPROCESS/ENCODE/SAMPLE/DECODE/INSERTING/DONE/ERROR/CANCELLED.
`JobConfig` (frozen): image_path, seed=42, steps=20, guidance_scale=3.0, shift=3.0, num_gaussians=262144, erode_radius=1, reuse_latent, placement(T,R,S), append(bool), output_format, output_path.
`JobResult`: success, elapsed_s, num_gaussians, gaussian (captured), node_name, error.

Two entry points:
- **`_run_full()`** — PREPARING→LOADING_MODEL→PREPROCESS→ENCODE→SAMPLE→DECODE→INSERTING. `preprocess_image → encode_image → sample_latent` (stash `_cached_latent` under signature `(seed, steps, guidance_scale, shift, image)`) → `decode_latent(latent, num_gaussians)`.
- **`decode_only(num_gaussians)`** — DECODE→INSERTING only (Learned-Density re-decode, near-instant).

**Cache invalidation:** invalidate on seed/steps/guidance/shift/image change; **keep** across `num_gaussians`. Panel auto-picks `decode_only` vs `_run_full`. Returned `Gaussian` kept alive for node lifetime (DLPack).

### 7.5 `core/direct_output.py` — insertion (verbatim half + axis remap)
`_to_lf` copied verbatim (detach→contiguous→`lf.Tensor.from_dlpack`, numpy float32 fallback; never clone-then-drop).
`insert_gaussian(gaussian, *, group_name, splat_name, replace=True)`:
1. `scene = lf.get_scene()`; bail if `None`/`not is_valid()`.
2. If `replace`: `try: scene.remove_node(group_name, keep_children=False) except: pass`. If append mode: pick the next free `TripoSplat_NN` group name instead.
3. `parent_id = scene.add_group(group_name)`.
4. Pull raw tensors (means, scales-linear, quats wxyz, DC colors, opacities-prob).
5. **Axis remap by us:** `_DEFAULT_TRANSFORM=[[1,0,0],[0,0,-1],[0,1,0]]` on means + quats (unit-tested).
6. **Activation inversion:** `scales = scales_linear.clamp(min=1e-8).log()`; `opacities = torch.logit(opacities_prob.clamp(1e-6,1-1e-6))`.
7. **SH layout:** `sh0=colors.unsqueeze(1).float()` `[N,1,3]`; `shN=zeros(N,0,3)`; `opacity=opacities.unsqueeze(1).float()` `[N,1]`.
8. `scene.add_splat(name=splat_name, means=..., sh0=..., shN=..., scaling=..., rotation=..., opacity=..., sh_degree=0, scene_scale=1.0, parent=parent_id)`.
9. `scene.notify_changed()`.

### 7.6 `core/export.py` (new)
`save(gaussian, path, fmt)` → `gaussian.save_ply(path)` / `gaussian.save_splat(path)`. "Save to disk" only.

---

## 8. Threading contract

**Deliberate divergence from the siblings, documented so a reviewer doesn't "fix" it.** HYWorld2/VGGT mutate the scene directly from the worker; TripoSplat takes the safer ml-sharp-spirit path:
- Worker runs **load/preprocess/encode/sample/decode** only.
- **Only the final insert** (`remove_node`+`add_group`+`add_splat`+`set_node_transform`+`notify_changed`+gizmo attach) is marshaled to the UI thread via verified `lf.ui.schedule_on_ui_thread(callback)`.
- The marshaled callback re-checks `scene.is_valid()` and compares `lf.get_scene_generation()` against the job-start generation (staleness guard).

`schedule_on_ui_thread`, `scene.is_valid()`, `lf.get_scene_generation()` all confirmed in stubs. Progress/log/stage uses the lock-guarded `JobResult` + `on_update` polling pattern.

---

## 9. Panel design

Hybrid RML/RCSS `TripoSplatPanel(lf.ui.Panel)` mirroring VGGT, with one departure: an embedded immediate-mode region for the matte preview (rendered as a tensor, not `<img>`).
**Class attrs:** `id="triposplat.main"`, `label="TripoSplat"`, `space=MAIN_PANEL_TAB`, `order=226`, `template=<dir>/main_panel.rml`, `height_mode=CONTENT`, `update_interval_ms=150`.
**Lifecycle:** `on_mount` (doc + section states); `on_bind_model` (`create_data_model("triposplat")` + bind/bind_func/bind_event); `on_update` (diff signatures, `handle.dirty(*group)`); `on_unmount` (cancel job, detach gizmo, remove model, `lf.ui.free_plugin_textures(PLUGIN_NAME)`).

### 9.1 Matte preview — verified mechanism (`ui.image_tensor`)
RML has **no image primitive** (zero `<img>` in the codebase; RmlUi caches textures by source string → stale pixels). So: RML has `<div id="im-root">` inside a `data-if="has_matte"` Preview section; the panel implements `draw(self, ui)` and renders `ui.image_tensor("triposplat_matte", self._matte_rgba, (w,h))` width-constrained via `ui.get_content_region_avail()`. GPU-direct, cached by label, no temp file. Temp-PNG `<img>` retained only as an explicit fallback.

### 9.2 Placement — verified gizmo UX (one bound editor)
T/R/S fields and `lf.TransformGizmo` are **one two-way-bound editor** of the node transform:
- Persist one `g = lf.TransformGizmo()`; `g.attach_to_node(splat_name, visualizer_world=False)` (local space). Header toggle sets `g.operation ∈ {translate,rotate,scale}`.
- **Gizmo→fields:** `g.set_on_change(cb)` + `g.set_on_end(cb)`; `d=lf.decompose_transform(g.matrix)` → write translation/euler/scale Signals (throttle ~30–60 Hz; commit precise value on `on_end`).
- **Fields→gizmo/node:** `m=lf.compose_transform(t,euler_deg,s)` → `lf.set_node_transform(splat_name, m)`; gizmo idle-sync repositions the handle; `lf.ui.request_redraw()` after.
- Fields are a **view of the matrix** (recompose on selection). `g.detach()` on unmount/selection clear; re-attach on target change.

### 9.3 Transform convention (pinned — column-major footgun)
Standardize on the module-level **flat 16-float COLUMN-MAJOR** family: `lf.compose_transform` → `lf.set_node_transform` → read back via `TransformGizmo.matrix` + `lf.decompose_transform`. **Never** feed a flat list into the 2-D `Scene.set_node_transform([4,4])` overload; never read row-major `Scene.get_world_transform` as column-major. Round-trip unit test.

### 9.4 Layout (RML, top→bottom)
1. title + subtitle. 2. Download banner (`data-if="model_downloading"`) + progress + cancel/retry. 3. **Input**: path + Browse (`open_image_dialog`). 4. **Preview** (`data-if="has_matte"`): `<div id="im-root">` + `erode_radius` slider. 5. **Settings** (collapsing): seed, steps (1–50), guidance (1.0–10.0), `num_gaussians` select {32768,65536,131072,262144}, advanced shift, **append-to-scene checkbox**; **Re-decode at count** button when latent cached. 6. **Model**: status + Load/Unload/Re-download. 7. **Placement** (`data-if="has_node"`): T/R/S rows + gizmo mode/space toggles. 8. **Run/Cancel** swap. 9. **Log** (`data-if="show_logs"`). 10. **Result**: count, elapsed, Save .ply/.splat. **Error** section.

---

## 10. Data flow

1. Pick image → UI-thread `lf.ui.open_image_dialog()` → validate ext.
2. Auto-preprocess (worker) → matte tensor → preview. erode_radius re-runs preprocess only.
3. Settings (clamped to verified ranges); append toggle decides replace vs accumulate.
4. Generate → `_run_full`: LOADING_MODEL→PREPROCESS(if stale)→ENCODE→SAMPLE(cache latent)→DECODE.
5. Insert (marshaled): re-check scene → `insert_gaussian` → `set_node_transform(compose_transform(placement))` → `notify_changed` → (re)attach gizmo. Replace-by-name unless append.
6. Re-decode at different count → `decode_only` (near-instant). Re-sample only on seed/steps/guidance/shift/image change.
7. Place: T/R/S ↔ gizmo (one editor).
8. Save to disk (optional).

---

## 11. Node naming + re-run behavior

- **Parent-group convention (HYWorld2):** group `"TripoSplat"` + child splat `"TripoSplat / splats"`.
- **Re-run = replace-by-name (default):** `remove_node("TripoSplat", keep_children=False)` before re-insert.
- **Append toggle:** when enabled, insert under the next free `"TripoSplat_NN"` group instead of replacing, so multiple generated objects coexist in one scene.
- **Undo:** none (matches all four siblings); replace-by-name is the de-facto undo.

---

## 12. Correctness contract (unit-tested)
- Activation inversion: `log()` scale, `logit()` opacity (LFS applies exp/sigmoid internally — double-activation = blown-out splat).
- SH layout `sh0 [N,1,3]` / `shN [N,0,3]`; opacity `[N,1]`; quaternions wxyz.
- Axis remap `_DEFAULT_TRANSFORM` applied by us (we capture raw tensors).
- DLPack ownership: keep source tensors alive for node lifetime; never clone-then-drop.
- Transform: column-major path only.
- Latent cache signature `(seed, steps, guidance, shift, image)`; reuse across count.
- bf16/compute guard: require CUDA + compute ≥ 8.0, else clear unsupported-hardware error.

---

## 13. Error handling
- No CUDA / compute < 8.0: clear panel error pre-load; never CPU.
- Missing/partial weights: `is_ready()` 5-file check → Download banner; cancellable; retry.
- OOM (unlikely at 24 GB): catch `OutOfMemoryError` → message + `empty_cache()`; suggest lower count.
- Stale scene at insert: marshaled callback aborts on None/invalid/generation mismatch.
- Stale gizmo node: `attach_to_node` no-ops; detach/re-attach on selection change.
- Windows stderr/tqdm EINVAL guard; verified TLS (no SSL-disable hack).
- Teardown sequence as §6.

---

## 14. Testing
`conftest.py` (verbatim from VGGT) patches `Package.setup` so tests collect without importing the lichtfeld-dependent root.
**Unit (GPU-free):** `test_direct_output_math.py` (inversion round-trip, SH shapes, quat order); `test_transform.py` (`_DEFAULT_TRANSFORM` value/orthonormality, compose→decompose round-trip, no 2-D overload); `test_job_cache.py` (cache invalidation, `decode_only` skips encode/sample, JobConfig immutability, append-name selection); `test_downloads.py` (5-file completeness, cancel state, progress math).
**GPU integration (`@pytest.mark.gpu`, opt-in):** end-to-end → non-empty splat node with finite means/opacity.
**Manual smoke:** load in LFS; generate into empty + existing scene; matte preview; re-decode; T/R/S↔gizmo sync; append toggle; Save round-trip.
**CI:** structure-integrity + `ruff check .` + `uv lock --no-cache` on ubuntu+windows.

---

## 15. Implementation order
1. Scaffold + packaging (tree, pyproject, .gitignore, LICENSE, NOTICE, conftest, install/uninstall, CI).
2. Vendor `triposplat.py`+`model.py` at pinned commit; **re-verify class/method names, the `Gaussian` accessors, the encode/sample/decode dict keys, and the 5 weight filenames against the vendored source** (the one flagged residual risk).
3. `__init__.py` load/teardown (copy sibling; swap names + HF cache; AppState subscribe in try/except; auto background download).
4. `core/downloads.py` + `core/pipeline_loader.py` (5-path ctor + bf16 guard).
5. `core/direct_output.py` (verbatim `_to_lf` + insertion; add axis remap) — TDD with `test_direct_output_math.py` + `test_transform.py` first.
6. `core/preprocess.py`, `core/job.py` (threading skeleton + latent cache + decode_only) + `test_job_cache.py`.
7. `core/export.py`.
8. Panel: VGGT scaffold; new fields (incl. append checkbox); `draw(ui)` matte via `ui.image_tensor`; gizmo two-way sync; placement + run/cancel + result.
9. Marshaled insert via `schedule_on_ui_thread` + staleness guard.
10. README/NOTICE finalize; GPU integration test; manual smoke.

---

## Appendix A — Upstream TripoSplat facts (research-derived; re-verify at vendoring)
- Single-image generative (flow-matching + CFG) → Gaussian splat. DINOv3 ViT-H/16 + Flux2 VAE encode → `LatentSeqMMFlowModel` (FlowEulerCfgSampler) → `OctreeGaussianDecoder`.
- **Learned Density Control:** one denoiser pass decodes at 32768/65536/131072/262144 by replaying only the decoder.
- Output: gaussians in normalized AABB `[-0.5,0.5]³` with `_DEFAULT_TRANSFORM=[[1,0,0],[0,0,-1],[0,1,0]]` applied on export; **no world coordinates**.
- Settings & defaults: seed; steps 1–50 (def 20); guidance_scale 1–10 (def 3.0); num_gaussians (def 262144); shift (def 3.0); erode_radius (def 1); output_format ply/splat.
- Preprocess: aspect resize → BiRefNet bg removal (if no real alpha) → alpha erosion `MinFilter(2r+1)` → bbox crop +20% pad → resize 1024² LANCZOS → composite on black.
- Weights: HF `VAST-AI/TripoSplat`, ~3.78 GB / 5 safetensors, ungated, MIT. Mixed dtype: bf16 encoders, fp16 flow+decoder. CUDA-only.
- Deps: torch, torchvision, numpy, safetensors, pillow, tqdm. License: code + weights MIT.

## Appendix B — Verification provenance
All load-bearing APIs are either confirmed in the LFS stubs (`schedule_on_ui_thread`, `scene.is_valid`/generation, `add_splat`/`add_group`/`remove_node`/`notify_changed`, `Tensor.from_dlpack`/`from_numpy`, `TransformGizmo` + `compose`/`decompose_transform`, `ui.image_tensor` + `free_plugin_textures`, `open_image_dialog`) or verified against shipped sibling source on disk (the `AppState` reconciliation, `_to_lf`, the activation-inversion block, `conftest.py`, `install.ps1`, HYWorld2 `pyproject`). The upstream TripoSplat contract is high-confidence from corroborating sources; the single residual risk is implementation-time drift in the vendored upstream (exact `Gaussian` accessors + encode/sample dict keys), re-verified at step 2.
