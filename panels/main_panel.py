from __future__ import annotations

import threading
from pathlib import Path

import lichtfeld as lf

from ..core import downloads, export, pipeline_loader, preprocess
from ..core.job import JobConfig, JobResult, TripoSplatJob, num_gaussians_valid

PLUGIN_NAME = "triposplat_plugin"

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

    def draw(self, ui):
        # Rendered into <div id="im-root">. Show the matte if present.
        if self._matte_rgb is None:
            return
        h, w = self._matte_rgb.shape[0], self._matte_rgb.shape[1]
        avail_w, _ = ui.get_content_region_avail()
        disp_w = max(64.0, min(float(avail_w), 320.0))
        disp_h = disp_w * (h / max(1, w))
        # verify: ui.image_tensor(label, lichtfeld.Tensor, (w,h)) per stubs
        # (ui/__init__.pyi:1650). Tensor.from_numpy is the documented numpy bridge
        # (lichtfeld/__init__.pyi:538); numpy is not auto-accepted, so convert here.
        ui.image_tensor("triposplat_matte", lf.Tensor.from_numpy(self._matte_rgb), (disp_w, disp_h))

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
        # verify: no generic save_file_dialog / .splat dialog in host stubs
        # (ui/__init__.pyi only ships save_ply/sog/spz/las/...). Drive both ply
        # and splat writers through save_ply_file_dialog (path/name picker only;
        # the actual writer is chosen by self.save_format in export.save).
        path = lf.ui.save_ply_file_dialog(default_name=f"triposplat.{self.save_format}")
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
            # verify: decompose_transform returns dict keys 'translation'/'euler'/'scale'
            # (lichtfeld/__init__.pyi:400 docs "translation, rotation, scale"; compose_transform's
            # euler_deg param implies 'euler'). Confirm key names against the running host.
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
        # Write in the SAME visualizer-world frame the gizmo uses, so typing in
        # the fields and dragging the gizmo agree (not the legacy data-world
        # lf.set_node_transform).
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
        for name in ("settings",):
            content = self._doc.get_element_by_id(f"sec-{name}")
            arrow = self._doc.get_element_by_id(f"arrow-{name}")
            if content:
                content.set_class("collapsed", name in self._collapsed)
            if arrow:
                arrow.set_class("is-expanded", name not in self._collapsed)
