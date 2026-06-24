"""Tests for SelectionProgress used during strategy select_samples."""

from __future__ import annotations

from active_learning.core.selection import SelectionProgress


def test_selection_progress_monotonic_within_span() -> None:
    events: list[tuple[int, int, str | None]] = []

    def sink(c: int, t: int, m: str | None) -> None:
        events.append((c, t, m))

    root = SelectionProgress(sink)
    span = root.span(0, 50)
    span.step(0, 10, "a")
    span.step(5, 10, "b")
    span.step(10, 10, "c")
    assert events == [(0, 100, "a"), (25, 100, "b"), (50, 100, "c")]
    assert events[0][0] <= events[1][0] <= events[2][0]


def test_subspan_fractions() -> None:
    events: list[int] = []

    def sink(c: int, _t: int, _m: str | None) -> None:
        events.append(c)

    root = SelectionProgress(sink)
    outer = root.span(10, 80)
    inner = outer.subspan(0.5, 1.0)
    inner.step(1, 1, "done")
    assert events == [90]
