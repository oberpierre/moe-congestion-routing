"""Shared path helpers used across data prep and training configs."""

import os


def expand_path(value: str) -> str:
    """Expand ``$VAR``/``${VAR}`` and ``~`` in a path, failing loud if any var is unresolved."""
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "$" in expanded:
        raise ValueError(
            f"unresolved environment variable in path {value!r} (expanded to {expanded!r}); "
            "set it (e.g. in config.sh) before loading the config"
        )
    return expanded
