"""YAML ``extends``-chain resolution shared across config schemas."""

from pathlib import Path

import yaml

# Key a config file uses to name its base config(s). The loader consumes it and it is never a
# config dataclass field, so it is stripped before the dataclass is constructed.
_EXTENDS_KEY = "extends"


def load_yaml_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    """Load a yaml mapping, resolving an optional ``extends:`` chain into one merged dict.

    Bases are merged first (in listed order, each recursively resolved), then the current
    file's own keys override them. ``extends`` paths are relative to the file that declares
    them. Cycles raise rather than recurse forever.
    """
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, path))
        raise ValueError(f"circular config extends chain: {chain}")

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a valid yaml mapping, got {type(data).__name__}")

    bases = data.pop(_EXTENDS_KEY, None)
    if bases is None:
        return data

    base_paths = [bases] if isinstance(bases, str) else bases
    merged: dict = {}
    for base in base_paths:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged.update(load_yaml_with_extends(base_path, (*_seen, path)))
    merged.update(data)  # this file's own keys win over everything it extends
    return merged
