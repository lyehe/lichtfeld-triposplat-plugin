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
