"""Tests for minimum time between images (ID-based, milliseconds)."""

from active_learning.core.time_gap import (
    apply_min_milliseconds_between_images,
    filter_ids_by_min_milliseconds_between_images,
)
from active_learning.core.types import SelectionResult


def _id_at(hour: int, minute: int = 0) -> str:
    return f"pre/2025-01-01T{hour:02d}_{minute:02d}_00_000Z.png"


def test_filter_ids_noop_when_zero():
    ids = [_id_at(10), _id_at(10, 1)]
    out = filter_ids_by_min_milliseconds_between_images(ids, 0.0)
    assert out is ids


def test_filter_ids_keeps_order_and_spacing():
    a, b, c = _id_at(10, 0), _id_at(10, 5), _id_at(12)
    ids = [a, b, c]
    out = filter_ids_by_min_milliseconds_between_images(ids, 600_000.0)
    assert out == [a, c]


def test_apply_millis_prunes_result_metadata():
    a, b, c = _id_at(10, 0), _id_at(10, 5), _id_at(12)
    r = SelectionResult(
        selected_ids=[a, b, c],
        scores={a: 1.0, b: 2.0, c: 3.0},
    )
    out = apply_min_milliseconds_between_images(
        r,
        min_milliseconds=600_000.0,
    )
    assert out.selected_ids == [a, c]
    assert out.scores == {a: 1.0, c: 3.0}
    assert out.details["min_milliseconds_between_images"]["pruned"] == 1


def test_unparseable_id_does_not_block_spacing_in_filter():
    a, b = _id_at(10), "no-timestamp-here.png"
    out = filter_ids_by_min_milliseconds_between_images(
        [a, b],
        3_600_000.0,
    )
    assert out == [a, b]
