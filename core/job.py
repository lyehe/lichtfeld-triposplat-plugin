from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum

try:
    import lichtfeld as lf
except Exception:  # pragma: no cover - outside LFS host
    lf = None  # type: ignore[assignment]

from . import insertion, pipeline_loader, preprocess

_NUM_GAUSSIANS_MIN = 32768
_NUM_GAUSSIANS_MAX = 262144
_GAUSSIANS_PER_POINT = 32
# Upper bound on how long the worker waits for the UI-thread scene insertion
# (PLY write + lf.io.load + add_splat at up to 262k gaussians) to complete.
_INSERT_TIMEOUT_S = 120.0


def num_gaussians_valid(n: int) -> int:
    n = max(_NUM_GAUSSIANS_MIN, min(_NUM_GAUSSIANS_MAX, int(n)))
    if n % _GAUSSIANS_PER_POINT != 0:
        # Round half-up to the documented contract ((n % 32) >= 16 rounds up),
        # not Python round()'s round-half-to-even.
        n = ((n + _GAUSSIANS_PER_POINT // 2) // _GAUSSIANS_PER_POINT) * _GAUSSIANS_PER_POINT
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
        finally:
            # The pipeline stays resident across runs, so torch's CUDA caching
            # allocator keeps each run's transient peak (encoder features, the
            # per-step sampler activations, the octree-decoder workspace) and
            # never returns it to the OS -> a second run sees much less free VRAM
            # and can fragment/OOM. Trim it here. This frees only cached-but-unused
            # blocks; the live latent + gaussian we hold are untouched.
            self._trim_cuda_cache()

    @staticmethod
    def _trim_cuda_cache() -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best-effort cleanup; never fail a job on this
            pass

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
        # The marshalled callback has no return/raise channel back to this
        # worker, so capture both the result and any failure in node_holder
        # and re-surface them here after the wait.
        node_holder: dict = {}
        done = threading.Event()

        def _ui_insert():
            try:
                node_holder["name"] = insertion.insert_gaussian(
                    self._gaussian, append=cfg.append, log=self._log_line)
            except Exception as exc:  # noqa: BLE001 - re-raised on the worker thread
                node_holder["error"] = exc
            finally:
                done.set()

        lf.ui.schedule_on_ui_thread(_ui_insert)  # verify: stub signature
        # Budget: ~18 MB PLY write + lf.io.load + add_splat at 262k gaussians,
        # plus time spent behind a busy UI thread, on slow disks.
        if not done.wait(timeout=_INSERT_TIMEOUT_S):
            raise RuntimeError(
                f"Scene insertion did not complete within {_INSERT_TIMEOUT_S:.0f}s "
                "(UI thread busy)."
            )
        if "error" in node_holder:
            raise node_holder["error"]

        with self._lock:
            self._result = JobResult(
                True, elapsed_s=time.time() - t0, num_gaussians=count,
                node_name=node_holder.get("name") or "")
        self._set(JobStage.DONE, 1.0, "Done")
