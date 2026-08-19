import subprocess
import sys

import pytest

from moe_congestion_routing.probe_windows import parse_windows


def test_parse_windows_parses_inclusive_ranges():
    assert parse_windows(["0:500", "3200:3450"]) == [(0, 500), (3200, 3450)]


def test_parse_windows_empty_list():
    assert parse_windows([]) == []


def test_parse_windows_rejects_malformed_spec():
    with pytest.raises(ValueError, match="malformed probe window"):
        parse_windows(["0-500"])


def test_parse_windows_rejects_non_integer_bounds():
    with pytest.raises(ValueError, match="malformed probe window"):
        parse_windows(["a:b"])


def test_parse_windows_rejects_inverted_range():
    with pytest.raises(ValueError, match="start > end"):
        parse_windows(["500:0"])


def test_probe_windows_module_imports_no_torch():
    # A package-root module must stay importable from a login node with no GPU, so importing it
    # must not drag in torch. Run in a subprocess, since in-process torch may already have been
    # loaded by another test module in this session.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import moe_congestion_routing.probe_windows; "
            "assert 'torch' not in sys.modules, sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
