import sys

import pytest

from moe_congestion_routing.training.megatron_path import (
    MegatronLMNotVendoredError,
    ensure_on_path,
    megatron_root,
)


def test_megatron_root_returns_dir_when_present(tmp_path):
    root = tmp_path / "Megatron-LM"
    root.mkdir()
    assert megatron_root(root) == root


def test_megatron_root_raises_when_missing(tmp_path):
    with pytest.raises(MegatronLMNotVendoredError, match="git submodule update --init"):
        megatron_root(tmp_path / "Megatron-LM")


@pytest.fixture
def _restore_sys_path():
    before = list(sys.path)
    yield
    sys.path[:] = before


def test_adds_directory_to_sys_path_when_it_exists(tmp_path, _restore_sys_path):
    root = tmp_path / "Megatron-LM"
    root.mkdir()

    ensure_on_path(root)

    assert str(root) in sys.path


def test_does_not_duplicate_an_already_present_path(tmp_path, _restore_sys_path):
    root = tmp_path / "Megatron-LM"
    root.mkdir()

    ensure_on_path(root)
    ensure_on_path(root)

    assert sys.path.count(str(root)) == 1


def test_raises_and_leaves_sys_path_untouched_when_directory_is_missing(
    tmp_path, _restore_sys_path
):
    root = tmp_path / "Megatron-LM"  # deliberately never created
    before = list(sys.path)

    with pytest.raises(MegatronLMNotVendoredError, match="git submodule update --init"):
        ensure_on_path(root)

    assert sys.path == before


def test_megatron_core_is_actually_usable_once_on_path():
    """Minimal end-to-end smoke test confirms the vendored submodule is importable and produces
    correct results, not just that some path gets added to sys.path.
    """
    pytest.importorskip("triton", reason="megatron.core requires triton, unavailable on macOS")
    try:
        ensure_on_path()
    except MegatronLMNotVendoredError as e:
        pytest.skip(str(e))

    megatron_core_utils = pytest.importorskip("megatron.core.utils")

    assert megatron_core_utils.divide(10, 2) == 5
    with pytest.raises(AssertionError):
        megatron_core_utils.divide(10, 3)
