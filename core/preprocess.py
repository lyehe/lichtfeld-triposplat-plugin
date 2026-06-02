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
