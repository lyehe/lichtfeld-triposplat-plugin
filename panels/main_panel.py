from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import lichtfeld as lf

from ..core import downloads, pipeline_loader, preprocess
from ..core.job import JobConfig, JobResult, TripoSplatJob, num_gaussians_valid

PLUGIN_NAME = "triposplat_plugin"
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
_PREVIEW_DIR = PLUGIN_ROOT / "cache" / "previews"
_RML_PATH_SAFE_CHARS = "/:._-~"
_MIN_SCALE = 0.001  # never let a splat collapse to 0/degenerate scale (unrecoverable basis)

_DIRTY_DL = ("model_downloading", "model_error", "dl_progress_value", "dl_progress_pct",
             "dl_bytes_line", "dl_error_text", "model_status_line", "model_loaded",
             "confirm_redownload", "can_run")
_DIRTY_PREVIEW = ("has_matte",)
_DIRTY_RUN = ("stage_text", "progress_value", "progress_pct", "progress_status")
_DIRTY_RUNNING = ("show_idle", "show_running", "can_run")
_DIRTY_LOG = ("show_logs", "live_log_text")
_DIRTY_RESULT = ("show_results", "show_error", "error_text", "result_count",
                 "result_time", "has_node", "has_latent")
_DIRTY_PLACE = ("tx", "ty", "tz", "rx", "ry", "rz", "scl", "gizmo_mode", "gizmo_active", "edit_target")


def _encode_rml_path(path: Path | str) -> str:
    return quote(str(path), safe=_RML_PATH_SAFE_CHARS)


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


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
        self.append_mode = True
        # placement
        self.tx = self.ty = self.tz = 0.0
        self.rx = self.ry = self.rz = 0.0
        self.scl = 1.0
        self.gizmo_mode = "translate"
        # runtime
        self._job: TripoSplatJob | None = None
        self._last_result: JobResult | None = None
        self._matte_pil = None
        self._matte_rgb = None           # np.uint8 [H,W,3] used by inference/cache tests
        self._preview_token = uuid4().hex
        self._preprocess_generation = 0
        self._matte_preview_path: Path | None = None
        self._matte_preview_decorator = "none"
        self._matte_preview_size = (0, 0)
        self._matte_preview_applied = ""
        self._matte_preview_applied_size: tuple[int, int] | None = None
        self._preprocessed_id = ""
        self._cached_latent = None
        self._cached_signature = None
        self._node_name = ""
        self._generated_nodes = []   # every splat this plugin has inserted, for selection-follow
        self._last_selection = None  # last host selection seen (diff cache)
        self._gizmo = None
        self._confirm_redownload = False
        self._collapsed = {"advanced", "precise"}
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
        self._apply_matte_preview(doc)

    def on_unmount(self, doc):
        if self._job and self._job.is_running():
            self._job.cancel()
        self._detach_gizmo()
        _safe_unlink(self._matte_preview_path)
        self._matte_preview_path = None
        try:
            lf.ui.free_plugin_textures(PLUGIN_NAME)  # verify: stub
        except Exception:
            pass
        doc.remove_data_model("triposplat")
        self._doc = None
        self._handle = None

    def on_scene_changed(self, doc):
        del doc
        # A node was added/removed/edited. If the splat we're placing was deleted,
        # drop the stale gizmo + "Editing" target so the panel never points at a dead
        # node, and prune deleted nodes from the follow-selection list. Skipped while
        # a job runs so we never race the just-inserted node before its on-UI-thread
        # insert completes.
        return self._prune_dead_nodes()

    def _prune_dead_nodes(self):
        if self._is_running():
            return False
        alive = []
        for name in self._generated_nodes:
            try:
                exists = lf.get_node_visualizer_world_transform(name) is not None
            except Exception:  # noqa: BLE001
                exists = True  # uncertain -> keep it
            if exists:
                alive.append(name)
        changed = len(alive) != len(self._generated_nodes)
        self._generated_nodes = alive
        if self._node_name and self._node_name not in alive:
            self._detach_gizmo()
            self._node_name = ""
            self._last_selection = None
            self._dirty(*_DIRTY_PLACE, "has_node")
            changed = True
        return changed

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("triposplat")
        if model is None:
            return
        # two-way scalar bindings
        model.bind("image_path", lambda: self.image_path, self._set_image_path)
        model.bind_func("image_name", lambda: Path(self.image_path).name if self.image_path else "No image selected")
        model.bind("seed", lambda: str(self.seed), lambda v: self._set_int("seed", v, 0, 2**31 - 1))
        model.bind("steps", lambda: str(self.steps), lambda v: self._set_int("steps", v, 1, 50))
        model.bind("guidance_scale", lambda: f"{self.guidance_scale:.1f}",
                   lambda v: self._set_float("guidance_scale", v, 1.0, 10.0))
        model.bind("shift", lambda: f"{self.shift:.1f}", lambda v: self._set_float("shift", v, 1.0, 6.0))
        model.bind("num_gaussians", lambda: str(self.num_gaussians),
                   lambda v: self._set_int("num_gaussians", v, 32768, 262144))
        model.bind("erode_radius", lambda: str(self.erode_radius), self._set_erode)
        model.bind("append_mode", lambda: self.append_mode, lambda v: self._set_bool("append_mode", v))
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
        model.bind_func("confirm_redownload", lambda: self._confirm_redownload)
        model.bind_func("has_matte", lambda: self._matte_rgb is not None)
        model.bind_func("has_latent", lambda: self._cached_latent is not None)
        model.bind_func("has_node", lambda: bool(self._node_name))
        model.bind_func("edit_target", lambda: self._node_name or "(none)")
        model.bind_func("gizmo_active", lambda: self._gizmo is not None)
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
        model.bind_event("unload_model", lambda *_: threading.Thread(target=pipeline_loader.unload, daemon=True).start())
        model.bind_event("ask_redownload", self._on_ask_redownload)
        model.bind_event("confirm_redownload_yes", self._on_redownload_yes)
        model.bind_event("confirm_redownload_no", self._on_redownload_no)
        model.bind_event("finalize_placement", self._on_finalize_placement)
        model.bind_event("start_placement", self._on_start_placement)
        model.bind_event("do_start", self._on_start)
        model.bind_event("do_cancel", self._on_cancel)
        model.bind_event("redecode", self._on_redecode)
        self._handle = model.get_handle()

    def draw(self, ui):
        # RML panels receive RmlUILayout here; image_tensor is only available on
        # immediate-mode layouts. The matte is rendered through the RML element
        # decorator in _apply_matte_preview().
        del ui

    def _save_matte_preview(self, matte_pil, generation: int) -> None:
        old_path = self._matte_preview_path
        path = _PREVIEW_DIR / f"matte_preview_{self._preview_token}_{generation}.png"
        try:
            _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            matte_pil.save(path)
        except Exception as exc:  # noqa: BLE001
            lf.log.warn(f"[triposplat] matte preview save failed: {exc}")
            return

        self._matte_preview_path = path
        self._matte_preview_decorator = f"image({_encode_rml_path(path)})"
        self._matte_preview_size = tuple(int(v) for v in matte_pil.size)
        self._matte_preview_applied = ""
        self._matte_preview_applied_size = None
        if old_path != path:
            _safe_unlink(old_path)

    def _apply_matte_preview(self, doc) -> bool:
        if doc is None:
            return False
        root = doc.get_element_by_id("im-root")
        if not root:
            self._matte_preview_applied = ""
            self._matte_preview_applied_size = None
            return False

        changed = False
        decorator = self._matte_preview_decorator if self._matte_preview_path else "none"
        if self._matte_preview_applied != decorator:
            root.set_property("decorator", decorator)
            self._matte_preview_applied = decorator
            changed = True

        w, h = self._matte_preview_size
        if decorator == "none" or w <= 0 or h <= 0:
            return changed

        disp_w = int(round(max(64.0, min(float(w), 320.0))))
        disp_h = int(round(max(1.0, disp_w * (h / max(1, w)))))
        size = (disp_w, disp_h)
        if self._matte_preview_applied_size != size:
            root.set_property("width", f"{disp_w}dp")
            root.set_property("height", f"{disp_h}dp")
            self._matte_preview_applied_size = size
            changed = True

        return changed

    def on_update(self, doc):
        dirty = False
        preview_dirty = self._apply_matte_preview(doc)
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
        if self._sync_selection():
            dirty = True
        # LFS renders on demand, so when the app is otherwise idle on_update stops
        # being called and the viewport stops repainting. While a splat is selected
        # (so we may need to follow it) or the gizmo is active (placement in
        # progress), keep requesting frames so selection-follow runs promptly and
        # the viewport-overlay gizmo re-renders at the new node without needing a
        # stray input event. The loop idles again once nothing is selected.
        try:
            if self._gizmo is not None or (self._generated_nodes and lf.has_selection()):
                lf.ui.request_redraw()
        except Exception:  # noqa: BLE001
            pass
        return dirty or preview_dirty

    @staticmethod
    def _result_key(r):
        if r is None:
            return None
        return (r.success, r.num_gaussians, round(r.elapsed_s, 2), r.node_name, r.error)

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
        st = downloads.get_state()
        if st["stage"] == "downloading":
            return st.get("message", "Downloading weights...")
        if st["stage"] == "error":
            return "Weight download failed - re-download under Settings."
        if not downloads.is_ready():
            return "Weights download on first use."
        if pipeline_loader.is_loaded():
            return "Ready - model in VRAM"
        return "Ready - weights cached (3.8 GB)"

    # --- input ---
    def _set_image_path(self, v):
        self.image_path = str(v)
        self._kick_preprocess()

    def _on_browse_image(self, *_):
        path = lf.ui.open_image_dialog("")  # verify: stub returns '' on cancel
        if path:
            self.image_path = path
            self._dirty("image_path", "image_name")
            self._kick_preprocess()

    def _set_erode(self, v):
        self._set_int("erode_radius", v, 0, 8)
        self._kick_preprocess()

    def _kick_preprocess(self):
        """Run preprocess on a daemon thread; result feeds the matte preview."""
        if not self.image_path:
            return
        # Weights download on first use (image selection); idempotent.
        downloads.start_background_download()
        if not downloads.is_ready():
            return
        path, erode = self.image_path, self.erode_radius
        self._preprocess_generation += 1
        generation = self._preprocess_generation

        def _task():
            try:
                from PIL import Image
                pipe = pipeline_loader.get_pipeline()
                src = Image.open(path).convert("RGB")
                matte_pil, matte_rgb = preprocess.run_preprocess(pipe, src, erode)
                if generation != self._preprocess_generation:
                    return
                self._matte_pil, self._matte_rgb = matte_pil, matte_rgb
                self._preprocessed_id = f"{path}|{erode}"
                self._save_matte_preview(matte_pil, generation)
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
        # Only hand the cached matte/latent to the job if they match the CURRENT
        # image+erode. Otherwise a new image was just picked and its (async)
        # preprocess hasn't finished yet, so the cached matte is stale -> the job
        # must re-run preprocess for cfg.image_path, or it would regenerate the
        # OLD asset from the previous image's matte.
        matte_current = self._preprocessed_id == f"{self.image_path}|{self.erode_radius}"
        reuse = reuse_latent and matte_current
        job = TripoSplatJob(
            cfg,
            cached_latent=self._cached_latent if reuse else None,
            cached_signature=self._cached_signature if reuse else None,
            matte_pil=self._matte_pil if matte_current else None,
            preprocessed_id=self._preprocessed_id if matte_current else "",
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
            if self._node_name and self._node_name not in self._generated_nodes:
                self._generated_nodes.append(self._node_name)
            self._reset_placement_fields()
            self._attach_gizmo()

    def _on_ask_redownload(self, *_):
        self._confirm_redownload = True
        self._dirty(*_DIRTY_DL)

    def _on_redownload_no(self, *_):
        self._confirm_redownload = False
        self._dirty(*_DIRTY_DL)

    def _on_redownload_yes(self, *_):
        self._confirm_redownload = False
        downloads.delete_models()
        downloads.start_background_download()
        self._dirty(*_DIRTY_DL)

    # --- placement / gizmo ---
    def _attach_gizmo(self):
        if not self._node_name:
            return
        self._detach_gizmo()
        try:
            self._gizmo = lf.TransformGizmo()  # verify: stub ctor + attach/operation API
            self._gizmo.operation = self.gizmo_mode
            # Attach in visualizer-world space (the frame the viewport renders and
            # the user interacts with). visualizer_world defaults to True; passing
            # False drives the LEGACY data-world transform, which differs from the
            # visualizer frame by the 3DGS Y-down/Z-forward <-> viewer Y-up
            # convention (a flip of 2 of 3 axes) -> gizmo drag/rotate look reversed
            # in 2 dims. Keep the default.
            self._gizmo.attach_to_node(self._node_name)
            self._gizmo.set_on_change(self._on_gizmo_change)
            self._gizmo.set_on_end(self._on_gizmo_end)
            self._on_gizmo_change()  # sync T/R/S fields to the node's current transform
            lf.ui.request_redraw()   # repaint the viewport so the gizmo jumps to the new node now
        except Exception as exc:  # noqa: BLE001
            lf.log.warn(f"[triposplat] gizmo attach failed: {exc}")

    def _detach_gizmo(self):
        if self._gizmo is not None:
            try:
                self._gizmo.detach()
            except Exception:
                pass
            self._gizmo = None

    def _on_finalize_placement(self, *_):
        # Disable the gizmo and lock in the current placement (T/R/S fields stay editable).
        self._detach_gizmo()
        self._dirty(*_DIRTY_PLACE)

    def _on_start_placement(self, *_):
        # Re-attach the interactive gizmo to keep adjusting placement.
        self._attach_gizmo()
        self._dirty(*_DIRTY_PLACE)

    def _sync_selection(self):
        """Follow the scene: when one of THIS plugin's generated splats is selected
        in the viewport, retarget the placement gizmo + T/R/S fields to it. Polled
        once per frame from on_update; gated on a selection change so it only acts on
        transitions. Other scene nodes (e.g. a loaded training scene) are ignored."""
        try:
            sel = lf.get_selected_node_name() if lf.has_selection() else ""
        except Exception:  # noqa: BLE001
            return False
        if sel == self._last_selection:
            return False
        self._last_selection = sel
        if sel and sel != self._node_name and sel in self._generated_nodes:
            self._node_name = sel
            self._attach_gizmo()  # re-attaches + syncs T/R/S from the node's transform
            self._dirty(*_DIRTY_PLACE, "has_node")
            return True
        return False

    def _on_gizmo_change(self, *_):
        try:
            # decompose_transform returns keys translation / rotation_euler_deg / scale
            # (host module.cpp); any reflection is folded into scale.x.
            d = lf.decompose_transform(self._gizmo.matrix)
            self.tx, self.ty, self.tz = d["translation"]
            self.rx, self.ry, self.rz = d["rotation_euler_deg"]
            scl = abs(d["scale"][0])
            if scl < _MIN_SCALE:
                # Gizmo dragged the scale below the safe minimum -> clamp the node's
                # basis back up so it can't collapse to a degenerate, unrecoverable
                # transform (matches the typed-field clamp in _set_transform_field).
                self._clamp_node_scale_min()
                scl = _MIN_SCALE
            self.scl = scl
            self._dirty(*_DIRTY_PLACE)
        except Exception:
            pass

    def _on_gizmo_end(self, *_):
        self._on_gizmo_change()

    def _clamp_node_scale_min(self):
        """Rescale any basis column shorter than _MIN_SCALE back up to it, so a gizmo
        scale-to-zero can't leave the node with a degenerate (unrecoverable) basis."""
        try:
            m = list(lf.get_node_visualizer_world_transform(self._node_name))
        except Exception:  # noqa: BLE001
            return
        if not m or len(m) != 16:
            return
        changed = False
        for c in range(3):  # columns 0,1,2 are the x/y/z basis vectors
            ox, oy, oz = m[c * 4], m[c * 4 + 1], m[c * 4 + 2]
            length = (ox * ox + oy * oy + oz * oz) ** 0.5
            if length >= _MIN_SCALE:
                continue
            if length > 1e-12:
                k = _MIN_SCALE / length
                m[c * 4], m[c * 4 + 1], m[c * 4 + 2] = ox * k, oy * k, oz * k
            else:
                # fully collapsed column: rebuild it axis-aligned at the minimum length
                m[c * 4], m[c * 4 + 1], m[c * 4 + 2] = 0.0, 0.0, 0.0
                m[c * 4 + c] = _MIN_SCALE
            changed = True
        if changed:
            lf.set_node_visualizer_world_transform(self._node_name, m)

    def _set_gizmo_mode(self, v):
        self.gizmo_mode = str(v)
        if self._gizmo is not None:
            try:
                self._gizmo.operation = self.gizmo_mode
            except Exception:
                pass

    def _set_transform_field(self, axis, v):
        try:
            val = float(v)
        except (TypeError, ValueError):
            return
        if axis == "scl":
            # A 0 (or negative) scale collapses the basis columns to zero length, which
            # is degenerate and unrecoverable (can't scale back up from length 0), and
            # upsets the gizmo/renderer. Clamp to a tiny positive minimum.
            val = max(abs(val), _MIN_SCALE)
        setattr(self, axis, val)
        if not self._node_name:
            return
        # Operate on the node's CURRENT visualizer-world matrix (column-major 16).
        try:
            m = list(lf.get_node_visualizer_world_transform(self._node_name))
        except Exception:  # noqa: BLE001
            return
        if not m or len(m) != 16:
            return
        if axis == "scl":
            # Rescale each basis column to length |val|, keeping its DIRECTION (and
            # any reflection the visualizer-world frame carries). This avoids the
            # decompose -> euler -> recompose round-trip, where the host folds any
            # reflection into scale.x and re-extracts euler -- that round-trip flips
            # the model upside down on scale edits. Pure column math can't flip.
            target = abs(val)
            for c in range(3):  # columns 0,1,2 are the x/y/z basis vectors
                ox, oy, oz = m[c * 4], m[c * 4 + 1], m[c * 4 + 2]
                length = (ox * ox + oy * oy + oz * oz) ** 0.5
                if length > 1e-8:
                    k = target / length
                    m[c * 4], m[c * 4 + 1], m[c * 4 + 2] = ox * k, oy * k, oz * k
        elif axis in ("tx", "ty", "tz"):
            m[12 + {"tx": 0, "ty": 1, "tz": 2}[axis]] = val  # translation column
        elif axis in ("rx", "ry", "rz"):
            # Absolute euler needs a recompose; decompose to keep translation+scale,
            # set the one euler component, recompose.
            try:
                d = lf.decompose_transform(m)
                t, e, s = list(d["translation"]), list(d["rotation_euler_deg"]), list(d["scale"])
            except Exception:  # noqa: BLE001
                return
            e[{"rx": 0, "ry": 1, "rz": 2}[axis]] = val
            m = lf.compose_transform(t, e, s)
        # Write back in the SAME visualizer-world frame the gizmo uses, so typing in
        # the fields and dragging the gizmo agree (not the legacy data-world path).
        lf.set_node_visualizer_world_transform(self._node_name, m)
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
        for name in ("advanced", "precise"):
            content = self._doc.get_element_by_id(f"sec-{name}")
            arrow = self._doc.get_element_by_id(f"arrow-{name}")
            if content:
                content.set_class("collapsed", name in self._collapsed)
            if arrow:
                arrow.set_class("is-expanded", name not in self._collapsed)
