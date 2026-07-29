import pytest

from moe_congestion_routing.paths import expand_path


def test_expands_env_var_and_user(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_STORE", "/store/me")
    assert expand_path("${DATA_STORE}/datasets/x") == "/store/me/datasets/x"
    assert expand_path("$DATA_STORE/y") == "/store/me/y"


def test_leaves_plain_paths_untouched():
    assert expand_path("/abs/path") == "/abs/path"
    assert expand_path("relative/path") == "relative/path"


def test_fails_loud_on_unresolved_var(monkeypatch):
    monkeypatch.delenv("DATA_STORE", raising=False)
    with pytest.raises(ValueError, match="unresolved environment variable"):
        expand_path("${DATA_STORE}/x")
