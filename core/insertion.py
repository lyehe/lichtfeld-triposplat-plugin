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
    try:
        scene = lf.get_scene()
    except Exception as exc:  # noqa: BLE001 - mirror HYWorld2/VGGT: log + clean None
        _log(f"insertion: lf.get_scene() failed: {exc}")
        return None
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
        if sd is None:  # mirror ml-sharp: clean None instead of opaque AttributeError
            _log("insertion: lf.io.load returned no splat_data; skipping.")
            return None
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
