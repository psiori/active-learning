"""CRID-backed ID query helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import os

from interface.model.models import CRIDFilter
from active_learning.integrations.sama.seeded import (
    query_seeded_ids as _query_seeded_ids_from_sama,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CridPool:
    pool_ids: list[str]
    labeled_ids: list[str]
    seeded_ids_count: int = 0


class CridSource:
    """Queries CRID for candidate and labeled sample IDs."""

    def __init__(self, crid):
        self.crid = crid

    def query_ids(
        self,
        *,
        sensor_name,
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        max_size: int = 1000000,
        labeled_only: bool = False,
        exclude_crid_tags: list[str] | None = None,
    ) -> list[str]:
        query_filter = CRIDFilter()
        query_filter.sensor_name = sensor_name
        query_filter.start_datetime = start_datetime or datetime(2020, 1, 1)
        query_filter.end_datetime = end_datetime or datetime(2027, 1, 1)
        if labeled_only:
            query_filter.not_empty = ["label_*"]
        if exclude_crid_tags:
            query_filter.exclude_crid_tags = exclude_crid_tags
        dataset = self.crid.query(query_filter, max_size=max_size)
        if dataset is None or dataset.dataframe is None:
            return []
        return list(dataset.dataframe["blob"].tolist())

    def query_labeled_ids(self, **kwargs) -> list[str]:
        return self.query_ids(labeled_only=True, **kwargs)

    def query_seeded_ids(
        self,
        sama_project_id: int,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> set[str]:
        return _query_seeded_ids_from_sama(
            self.crid, sama_project_id, on_progress=on_progress
        )

    def query_all_ids_and_labeled_ids(
        self,
        *,
        sensor_name,
        exclude_crid_tags: list[str] | None = None,
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        max_size: int = 1000000,
    ) -> tuple[list[str], list[str]]:
        """Return (all candidate ids in range, labeled ids in range) from CRID."""
        all_ids = self.query_ids(
            sensor_name=sensor_name,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            max_size=max_size,
            exclude_crid_tags=exclude_crid_tags,
        )
        labeled_ids = self.query_labeled_ids(
            sensor_name=sensor_name,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            max_size=max_size,
        )
        return all_ids, labeled_ids

    def build_pool_with_seeded_exclusions(
        self,
        all_ids: list[str],
        labeled_ids: list[str],
        *,
        exclude_seeded: bool,
        sama_project_id: int | None = None,
        on_seeded_progress: Callable[[int, int], None] | None = None,
    ) -> CridPool:
        """Subtract labeled and (optionally) Sama-seeded ids from ``all_ids`` to form ``pool_ids``."""
        excluded_ids = set(labeled_ids)
        seeded_ids_count = 0
        if exclude_seeded:
            resolved_project_id = sama_project_id or os.environ.get("SAMA_PROJECT_ID")
            if resolved_project_id is None:
                raise ValueError(
                    "Sama project ID required to exclude seeded images. "
                    "Provide sama_project_id, set SAMA_PROJECT_ID, or disable exclude_seeded.",
                )
            seeded_ids = self.query_seeded_ids(
                int(resolved_project_id), on_progress=on_seeded_progress
            )
            seeded_ids_count = len(seeded_ids - excluded_ids)
            excluded_ids.update(seeded_ids)
        pool_ids = [sample_id for sample_id in all_ids if sample_id not in excluded_ids]
        return CridPool(
            pool_ids=pool_ids,
            labeled_ids=labeled_ids,
            seeded_ids_count=seeded_ids_count,
        )

    def query_pool_and_labeled_ids(
        self,
        *,
        sensor_name,
        sama_project_id: int | None = None,
        exclude_seeded: bool = True,
        exclude_crid_tags: list[str] | None = None,
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        max_size: int = 1000000,
        on_seeded_progress: Callable[[int, int], None] | None = None,
    ) -> CridPool:
        all_ids, labeled_ids = self.query_all_ids_and_labeled_ids(
            sensor_name=sensor_name,
            exclude_crid_tags=exclude_crid_tags,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            max_size=max_size,
        )
        return self.build_pool_with_seeded_exclusions(
            all_ids,
            labeled_ids,
            exclude_seeded=exclude_seeded,
            sama_project_id=sama_project_id,
            on_seeded_progress=on_seeded_progress,
        )
