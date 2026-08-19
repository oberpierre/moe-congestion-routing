"""The ``"start:end"`` window-spec grammar the router probe's dense schedule is written in."""


def parse_windows(specs: list[str]) -> list[tuple[int, int]]:
    """Parse inclusive ``"start:end"`` strings into integer ``(start, end)`` tuples.

    Raises ``ValueError`` naming the offending spec on a malformed string or an inverted range.
    """
    windows = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"malformed probe window {spec!r}, expected 'start:end'")
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"malformed probe window {spec!r}, expected 'start:end'") from exc
        if start > end:
            raise ValueError(f"probe window {spec!r} has start > end")
        windows.append((start, end))
    return windows
