import pytest

from moe_congestion_routing.config_extends import load_yaml_with_extends


def test_no_extends_returns_file_as_is(tmp_path):
    path = tmp_path / "plain.yaml"
    path.write_text("a: 1\nb: 2\n")
    assert load_yaml_with_extends(path) == {"a": 1, "b": 2}


def test_merges_base_first_then_own_keys_win(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb: 2\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\nb: 20\nc: 3\n")
    assert load_yaml_with_extends(child) == {"a": 1, "b": 20, "c": 3}


def test_list_valued_extends_merges_in_listed_order(tmp_path):
    # Later bases override earlier ones, and the file's own keys override every base.
    (tmp_path / "a.yaml").write_text("a: 1\nb: 1\nc: 1\n")
    (tmp_path / "b.yaml").write_text("b: 2\nc: 2\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: [a.yaml, b.yaml]\nc: 3\n")
    assert load_yaml_with_extends(child) == {"a": 1, "b": 2, "c": 3}


def test_extends_path_is_relative_to_declaring_file_not_cwd(tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "base.yaml").write_text("a: 1\n")
    child = sub / "child.yaml"
    child.write_text("extends: base.yaml\nb: 2\n")
    # cwd is the parent of `sub`, so a cwd-relative resolution of "base.yaml" would miss;
    # only resolving against child.yaml's own directory finds it.
    monkeypatch.chdir(tmp_path)
    assert load_yaml_with_extends(child) == {"a": 1, "b": 2}


def test_rejects_cycles(tmp_path):
    (tmp_path / "x.yaml").write_text("extends: y.yaml\n")
    (tmp_path / "y.yaml").write_text("extends: x.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_yaml_with_extends(tmp_path / "x.yaml")
