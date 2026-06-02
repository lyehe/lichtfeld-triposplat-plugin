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
