"""Filter sample IDs by minimum capture time (from paths only; no image download)."""

from __future__ import annotations

from copy import deepcopy

from active_learning.core.sample_id import datetime_from_sample_id
from active_learning.core.types import SelectionResult


def filter_ids_by_min_milliseconds_between_images(
    ids: list[str],
    min_milliseconds: float,
) -> list[str]:
    """Keep samples whose parseable times are at least *min_milliseconds* apart, chronologically.

    Parses times, sorts by capture time, one forward scan (keep if at least *min* ms
    after the last kept time),     then returns IDs in the original *ids* list order.

    IDs with no parseable time are always kept. O(n log n) for time sort; no per-insert
    list shuffles. If ES document order and time order differ, this rule (chronological)
    can differ from a strict “pool order / check all previous keeps” rule on rare
    border cases.

    No-op (returns *ids* unchanged) when *min_milliseconds* <= 0 or *ids* is empty.
    """
    if min_milliseconds <= 0.0 or not ids:
        return ids

    parsed: list[tuple[str, float]] = []
    no_ts: set[str] = set()
    for sid in ids:
        dt = datetime_from_sample_id(sid)
        if dt is None:
            no_ts.add(sid)
            continue
        t_ms = dt.timestamp() * 1000.0
        parsed.append((sid, t_ms))
    parsed.sort(key=lambda x: x[1])
    # Chronological greed: 1D, only need gap from the last *kept* time when times are sorted.
    keep_parseable: set[str] = set()
    last_kept: float | None = None
    for sid, t_ms in parsed:
        if last_kept is None or t_ms - last_kept >= min_milliseconds:
            keep_parseable.add(sid)
            last_kept = t_ms

    return [sid for sid in ids if sid in no_ts or sid in keep_parseable]


def apply_min_milliseconds_between_images(
    result: SelectionResult,
    *,
    min_milliseconds: float,
) -> SelectionResult:
    """Prune *result.selected_ids* and aligned *scores* / *artifacts* using
    :func:`filter_ids_by_min_milliseconds_between_images`.

    A disabled or non-positive *min_milliseconds* returns *result* unchanged.
    """
    if min_milliseconds <= 0.0 or not result.selected_ids:
        return result

    accepted = filter_ids_by_min_milliseconds_between_images(
        result.selected_ids,
        min_milliseconds,
    )
    n_pruned = len(result.selected_ids) - len(accepted)
    if n_pruned == 0:
        return result

    kept = set(accepted)
    new_scores = {k: v for k, v in result.scores.items() if k in kept}
    new_artifacts: dict = {}
    for key, mapping in result.artifacts.items():
        if isinstance(mapping, dict):
            new_artifacts[key] = {k: v for k, v in mapping.items() if k in kept}
        else:
            new_artifacts[key] = deepcopy(mapping)
    new_details = {
        **result.details,
        "min_milliseconds_between_images": {
            "min_milliseconds": min_milliseconds,
            "pruned": n_pruned,
            "before": len(result.selected_ids),
            "after": len(accepted),
        },
    }
    return SelectionResult(
        selected_ids=accepted,
        scores=new_scores,
        artifacts=new_artifacts,
        details=new_details,
    )
