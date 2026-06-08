# TripoSplat Plugin for LichtFeld Studio

Single image → 3D Gaussian splat, straight into your scene. Runs
[VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat) on one product
photo and inserts the generated Gaussian splat into the current LichtFeld Studio
scene — ready to view, place, train alongside, or export.

## Features

- **Single-image generation** — pick one RGB photo; the model produces a full
  3D Gaussian splat. No camera poses, no COLMAP, no multi-view capture.
- **Live matte preview** — the background-removal matte (RGB on black) renders
  in the panel before you generate, so you can confirm the cutout is clean first.
- **Inserted directly into the scene** — the splat lands at the origin of your
  current scene via `lf.io.load`, so it sits alongside whatever is already
  loaded. Generating again **replaces** the previous result by default, or
  **appends** a new `TripoSplat_01`, `TripoSplat_02`, … when Append is on.
- **Re-decode without re-sampling** — change the Gaussian count and the cached
  latent is decoded again in a near-instant pass; no need to re-run the
  diffusion sampler.
- **Place with a gizmo or fields** — a transform gizmo on the inserted node is
  two-way bound to Translate / Rotate / Scale fields: drag in the viewport and
  the fields update; type a value and the node + gizmo move.
- **Save to disk** — export the decoded Gaussian as `.ply` or `.splat`.
- **Auto-unload on training start** — when an LFS training run begins, the
  ~3.8 GB pipeline is released from VRAM so the trainer has room.

## Requirements

- **GPU**: CUDA-capable NVIDIA GPU with compute capability **≥ 8.0**
  (Ampere / RTX 30xx or newer) — the pipeline runs in bfloat16 and guards on
  compute capability at load.
- **Disk**: ~4 GB for the plugin venv + ~3.8 GB for the model weights
  (downloaded automatically on first load into `<plugin>/models/`).
- **VRAM**: enough headroom for the ~3.8 GB pipeline plus your scene.

## Installation

### Via LichtFeld UI (recommended)

1. Open LichtFeld Studio
2. Go to the **Plugins** panel
3. Paste the GitHub URL: `https://github.com/lyehe/lichtfeld-triposplat-plugin`
4. Click **Install**
5. Restart LichtFeld Studio

### Via Python

```python
import lichtfeld as lf
lf.plugins.install("lyehe/lichtfeld-triposplat-plugin")
```

### Manual (dev junction/symlink)

```powershell
# Windows — no admin; uses a directory junction
git clone https://github.com/lyehe/lichtfeld-triposplat-plugin
cd lichtfeld-triposplat-plugin
.\install.ps1
```

```bash
# Linux / macOS
git clone https://github.com/lyehe/lichtfeld-triposplat-plugin
cd lichtfeld-triposplat-plugin
./install.sh
```

On first load, LFS runs `uv sync` in the plugin dir to build the venv (CUDA
torch 2.11 cu130, torchvision, triton-windows on Windows, etc.); subsequent
launches are near-instant. The ~3.8 GB TripoSplat weights (5 safetensors files
from `VAST-AI/TripoSplat`) download lazily in the background on first load — the
panel shows a progress banner until they're ready. When you see
`triposplat_plugin loaded` in the log, the plugin is active.

## Usage

1. Open the **TripoSplat** panel in LichtFeld Studio.
2. Click **Browse** and select a single RGB image (a product photo on a clean
   background works best).
3. The **matte preview** (the background-removed cutout, RGB on black) renders
   in the panel, so you can confirm the cutout is clean before generating.
4. Expand **Settings** to tune seed / steps / guidance / shift / Gaussian count
   if you want (defaults are fine for a first run).
5. Click **Generate**. Progress runs through staged steps
   (preprocess → encode → sample → decode → inserting) and the splat appears at
   the origin of your current scene.
6. **Place it**: drag the transform gizmo in the viewport, or type into the
   Translate / Rotate / Scale fields — the two stay in sync. Switch the gizmo
   between translate / rotate / scale modes.
7. (Optional) Change the **Gaussian count** and click **Re-decode at this
   count** for a near-instant re-decode of the same latent.
8. (Optional) Click **Save to disk** to write the result as `.ply` or `.splat`.

To add a second object without clearing the first, enable **Append** before the
next Generate.

## Settings

Expand **Settings** in the panel. Defaults in parentheses.

| Setting | Range | Default | Effect |
|---|---|---|---|
| **Seed** | 0 – 2³¹−1 | 42 | RNG seed for the diffusion sampler; change for a different generation. |
| **Steps** | 1 – 50 | 20 | Diffusion sampling steps. More = slower, potentially cleaner. |
| **Guidance scale** | 1.0 – 10.0 | 3.0 | Classifier-free guidance strength. |
| **Shift** | 1.0 – 6.0 | 3.0 | Flow-matching timestep shift. |
| **Gaussians** | 32,768 – 262,144 | 262,144 | Output Gaussian count; rounded to a multiple of 32. Drives **Re-decode**. |

Other controls:

- **Append** (off by default) — when off, each Generate **replaces** the
  previous `TripoSplat` node; when on, it **appends** `TripoSplat_01`,
  `TripoSplat_02`, … so multiple objects coexist in the scene.
- **Re-decode at this count** — re-decodes the cached latent at the current
  Gaussian count without re-running the sampler. Only changing the seed / steps /
  guidance / shift (the sampling inputs) invalidates the cache and
  requires a full Generate.
- **Save format** — `ply` or `splat`, applied by **Save to disk**.

## Notes

- **Plugin code:** MIT — see [LICENSE](LICENSE).
- **Model weights:** the `VAST-AI/TripoSplat` weights are MIT-licensed and are
  downloaded at runtime from Hugging Face into `<plugin>/models/`. They are
  **not redistributed** with this plugin.
- **Vendored source:** the upstream TripoSplat source is vendored under
  `triposplat/` (MIT). See [NOTICE](NOTICE) for third-party attributions.

## Uninstall

```powershell
# Windows
.\uninstall.ps1
```

```bash
# Linux / macOS
./uninstall.sh
```

This removes the junction/symlink from `~/.lichtfeld/plugins/triposplat_plugin`.
Delete the plugin's `models/` and `cache/` directories to reclaim the downloaded
weights and the torch.compile / Triton caches.
