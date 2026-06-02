from core import insertion


class _FakeScene:
    def __init__(self, existing):
        self._names = set(existing)

    def has_node(self, name):
        return name in self._names


def test_next_group_name_replace_mode_returns_base():
    scene = _FakeScene({"TripoSplat", "TripoSplat_01"})
    assert insertion.next_group_name(scene, append=False) == "TripoSplat"


def test_next_group_name_append_picks_first_free_slot():
    scene = _FakeScene({"TripoSplat", "TripoSplat_01"})
    assert insertion.next_group_name(scene, append=True) == "TripoSplat_02"


def test_next_group_name_append_from_empty_scene():
    scene = _FakeScene(set())
    assert insertion.next_group_name(scene, append=True) == "TripoSplat_01"
