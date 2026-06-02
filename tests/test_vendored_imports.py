"""Regression guard for the vendored TripoSplat package's intra-package imports.

Task 2 adapted upstream's flat layout (top-level ``triposplat.py`` + ``model.py``
modules) into a *package* (``triposplat/`` dir + ``__init__.py``). Under the flat
layout ``from triposplat import _build_gaussians`` resolved to ``triposplat.py``.
Under the package layout that same absolute import resolves to
``triposplat/__init__.py`` -- which only re-exports ``TripoSplatPipeline`` and
``Gaussian`` -- so it raises ``ImportError: cannot import name '_build_gaussians'``.

``OctreeGaussianDecoder.decode()`` performs exactly this import on every
re-decode (``TripoSplatPipeline.decode_latent`` -> ``decoder.decode``), the core
path the whole plugin depends on. These tests pin the import to its
package-relative form and prove that form actually resolves at call time.

No torch / GPU required: a minimal in-memory mirror of the vendored package
structure reproduces the import-resolution behaviour faithfully.
"""

from __future__ import annotations

import ast
import importlib
import sys
import textwrap
from pathlib import Path

import pytest

_VENDOR_DIR = Path(__file__).resolve().parent.parent / "triposplat"


def _decode_import_node() -> ast.ImportFrom:
    """Return the ``from ... import _build_gaussians`` node inside decode()."""
    tree = ast.parse((_VENDOR_DIR / "model.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "decode":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.ImportFrom) and any(
                    alias.name == "_build_gaussians" for alias in stmt.names
                ):
                    return stmt
    raise AssertionError("no `import _build_gaussians` found inside decode()")


def test_decode_import_is_package_relative():
    """decode() must import _build_gaussians package-relatively, not absolutely.

    An absolute ``from triposplat import _build_gaussians`` (level == 0) hits
    ``__init__.py`` and fails; the fix is ``from .triposplat import ...``.
    """
    node = _decode_import_node()
    assert node.module == "triposplat", node.module
    assert node.level == 1, (
        "decode() uses an ABSOLUTE self-import (level 0); under the package "
        "layout this resolves to __init__.py and raises ImportError. It must be "
        "package-relative: `from .triposplat import _build_gaussians`."
    )


def test_no_absolute_triposplat_self_import_in_vendored_modules():
    """No vendored module may import itself via the absolute top-level name."""
    offenders: list[str] = []
    for py in ("model.py", "triposplat.py"):
        tree = ast.parse((_VENDOR_DIR / py).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in {
                "triposplat",
                "model",
            }:
                offenders.append(f"{py}: from {node.module} import ...")
    assert not offenders, (
        "absolute intra-package self-imports break under the package layout; "
        f"make them relative: {offenders}"
    )


def test_relative_import_actually_resolves_build_gaussians(tmp_path, monkeypatch):
    """Actually invoke a decode() whose import line mirrors the real source.

    Builds a torch-free mirror of the vendored package layout:
      pkg/__init__.py      -> re-exports only the public API (as the plan pins it)
      pkg/triposplat.py    -> defines _build_gaussians
      pkg/model.py         -> decode() does `from .triposplat import _build_gaussians`

    Calling decode() exercises the exact import the real bug lived in. With the
    absolute form this raises ImportError; with the relative form it resolves.
    """
    pkg = tmp_path / "ts_mirror"
    pkg.mkdir()
    # __init__.py mirrors the plan's pinned public export surface (no _build_gaussians).
    (pkg / "__init__.py").write_text(
        "from .triposplat import TripoSplatPipeline, Gaussian  # noqa: F401\n",
        encoding="utf-8",
    )
    (pkg / "triposplat.py").write_text(
        textwrap.dedent(
            """
            class TripoSplatPipeline:  # public API placeholder
                pass

            class Gaussian:  # public API placeholder
                pass

            def _build_gaussians(decoder, points_pred, pred):
                return ["built"]
            """
        ),
        encoding="utf-8",
    )
    (pkg / "model.py").write_text(
        textwrap.dedent(
            """
            class OctreeGaussianDecoder:
                def decode(self):
                    # mirrors triposplat/model.py decode() exactly
                    from .triposplat import _build_gaussians
                    return _build_gaussians(None, None, None)[0]
            """
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in list(sys.modules):
        if mod == "ts_mirror" or mod.startswith("ts_mirror."):
            del sys.modules[mod]

    model_mod = importlib.import_module("ts_mirror.model")
    # The import lives inside decode(); calling it is what triggers resolution.
    assert model_mod.OctreeGaussianDecoder().decode() == "built"


def test_absolute_form_would_fail(tmp_path, monkeypatch):
    """Negative control: the OLD absolute form raises ImportError, proving the bug."""
    pkg = tmp_path / "ts_mirror_bad"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from .triposplat import TripoSplatPipeline, Gaussian  # noqa: F401\n",
        encoding="utf-8",
    )
    (pkg / "triposplat.py").write_text(
        textwrap.dedent(
            """
            class TripoSplatPipeline:
                pass

            class Gaussian:
                pass

            def _build_gaussians(decoder, points_pred, pred):
                return ["built"]
            """
        ),
        encoding="utf-8",
    )
    (pkg / "model.py").write_text(
        textwrap.dedent(
            """
            class OctreeGaussianDecoder:
                def decode(self):
                    from ts_mirror_bad import _build_gaussians  # absolute -> __init__.py
                    return _build_gaussians(None, None, None)[0]
            """
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in list(sys.modules):
        if mod == "ts_mirror_bad" or mod.startswith("ts_mirror_bad."):
            del sys.modules[mod]

    model_mod = importlib.import_module("ts_mirror_bad.model")
    with pytest.raises(ImportError, match="_build_gaussians"):
        model_mod.OctreeGaussianDecoder().decode()
