from core import insertion


class _FakeScene:
    def __init__(self, existing):
        self._names = set(existing)

    def get_node(self, name):
        # Mirror the real Scene.get_node(name) -> SceneNode | None contract.
        return object() if name in self._names else None


def test_next_group_name_replace_mode_returns_base():
    scene = _FakeScene({"TripoSplat", "TripoSplat_01"})
    assert insertion.next_group_name(scene, append=False) == "TripoSplat"


def test_next_group_name_append_picks_first_free_slot():
    scene = _FakeScene({"TripoSplat", "TripoSplat_01"})
    assert insertion.next_group_name(scene, append=True) == "TripoSplat_02"


def test_next_group_name_append_from_empty_scene():
    scene = _FakeScene(set())
    assert insertion.next_group_name(scene, append=True) == "TripoSplat_01"
