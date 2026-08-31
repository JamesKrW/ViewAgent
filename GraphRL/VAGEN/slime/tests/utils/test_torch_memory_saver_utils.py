import types

import pytest

from slime.utils.torch_memory_saver_utils import _nested_region_config


class _FakeCdll:
    def __init__(self, *, active: bool, cpu_backup: bool):
        self.active = active
        self.cpu_backup = cpu_backup

    def tms_get_interesting_region(self):
        return self.active

    def tms_get_enable_cpu_backup(self):
        return self.cpu_backup


class _FakeWrapper:
    def __init__(self, cdll):
        self.cdll = cdll
        self.tag = "default"

    def set_config(self, *, tag, interesting_region, enable_cpu_backup):
        self.tag = tag
        self.cdll.active = interesting_region
        self.cdll.cpu_backup = enable_cpu_backup


@pytest.mark.unit
def test_nested_region_restores_active_preload_region():
    cdll = _FakeCdll(active=True, cpu_backup=True)
    impl = types.SimpleNamespace(_binary_wrapper=_FakeWrapper(cdll))

    with _nested_region_config(impl, "param_buffer", enable_cpu_backup=False):
        assert cdll.active is True
        assert cdll.cpu_backup is False
        assert impl._binary_wrapper.tag == "param_buffer"

    assert cdll.active is True
    assert cdll.cpu_backup is True
    assert impl._binary_wrapper.tag == "default"


@pytest.mark.unit
def test_nested_region_restores_an_outer_nested_tag():
    cdll = _FakeCdll(active=True, cpu_backup=True)
    impl = types.SimpleNamespace(_binary_wrapper=_FakeWrapper(cdll))

    with _nested_region_config(impl, "outer", enable_cpu_backup=False):
        with _nested_region_config(impl, "inner", enable_cpu_backup=True):
            assert impl._binary_wrapper.tag == "inner"
        assert impl._binary_wrapper.tag == "outer"
        assert cdll.cpu_backup is False

    assert impl._binary_wrapper.tag == "default"
    assert cdll.cpu_backup is True
