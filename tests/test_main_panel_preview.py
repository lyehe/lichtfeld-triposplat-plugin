from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from PIL import Image


class _ElementStub:
    def __init__(self):
        self.properties = {}

    def set_property(self, name, value):
        self.properties[name] = value


class _DocumentStub:
    def __init__(self, element):
        self._element = element

    def get_element_by_id(self, element_id):
        if element_id == "im-root":
            return self._element
        return None


def _install_lf_stub(monkeypatch):
    class _Panel:
        pass

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        Panel=_Panel,
        PanelSpace=SimpleNamespace(MAIN_PANEL_TAB="MAIN_PANEL_TAB"),
        PanelHeightMode=SimpleNamespace(CONTENT="CONTENT"),
        free_plugin_textures=lambda _plugin_name: None,
    )
    lf_stub.log = SimpleNamespace(
        warn=lambda _msg: None,
        info=lambda _msg: None,
        error=lambda _msg: None,
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)


def _import_panel_module(monkeypatch):
    root = Path(__file__).resolve().parent.parent
    package_name = "triposplat_plugin"
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]

    package = ModuleType(package_name)
    package.__file__ = str(root / "__init__.py")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package_name, package)
    return importlib.import_module(f"{package_name}.panels.main_panel")


def test_draw_does_not_call_missing_rml_image_tensor(monkeypatch):
    _install_lf_stub(monkeypatch)
    main_panel = _import_panel_module(monkeypatch)
    panel = main_panel.TripoSplatPanel()

    panel.draw(SimpleNamespace())


def test_matte_preview_uses_encoded_rml_decorator(monkeypatch, tmp_path):
    _install_lf_stub(monkeypatch)
    main_panel = _import_panel_module(monkeypatch)
    preview_dir = tmp_path / "preview cache"
    monkeypatch.setattr(main_panel, "_PREVIEW_DIR", preview_dir)

    panel = main_panel.TripoSplatPanel()
    panel._preview_token = "test"
    panel._save_matte_preview(Image.new("RGB", (100, 50), "red"), generation=7)

    assert panel._matte_preview_path == preview_dir / "matte_preview_test_7.png"
    assert panel._matte_preview_path.exists()
    assert panel._matte_preview_decorator.startswith("image(")
    assert "%20" in panel._matte_preview_decorator

    element = _ElementStub()
    assert panel._apply_matte_preview(_DocumentStub(element)) is True
    assert element.properties["decorator"] == panel._matte_preview_decorator
    assert element.properties["width"] == "100dp"
    assert element.properties["height"] == "50dp"
