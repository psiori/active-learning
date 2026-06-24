"""Selector helper functions."""

from __future__ import annotations

import numpy as np

from active_learning.core.types import SampleId


def ids_with_artifacts(
    sample_ids: list[SampleId],
    artifact: dict[SampleId, object],
) -> list[SampleId]:
    """Preserving *sample_ids* order, keep only IDs present in *artifact*."""
    return [sid for sid in sample_ids if sid in artifact]


def matrix_for_ids(
    artifact: dict[SampleId, np.ndarray],
    sample_ids: list[SampleId],
) -> np.ndarray:
    return np.stack(
        [np.asarray(artifact[sample_id]) for sample_id in sample_ids],
        axis=0,
    )


def vector_for_ids(
    artifact: dict[SampleId, float],
    sample_ids: list[SampleId],
) -> np.ndarray:
    return np.asarray(
        [float(artifact[sample_id]) for sample_id in sample_ids],
        dtype=np.float32,
    )
