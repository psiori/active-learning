"""Composite sink that exports to CRID and submits a Sama batch."""

from __future__ import annotations

from dataclasses import dataclass

from active_learning.integrations.crid.export import export_selection, get_export_urls
from active_learning.integrations.sama.seed import create_batch
from active_learning.core.types import SelectionResult


@dataclass(frozen=True, slots=True)
class CridSamaSinkResult:
    export_id: str
    description: str
    sama_batch_id: str | None
    image_count: int


class CridSamaSink:
    def __init__(
        self,
        crid,
        *,
        sama_project_id: int,
        priority: int = 0,
        overwrite: bool = False,
    ):
        self.crid = crid
        self.sama_project_id = sama_project_id
        self.priority = priority
        self.overwrite = overwrite

    def submit(
        self,
        result: SelectionResult,
        *,
        export_id: str,
        description: str,
    ) -> CridSamaSinkResult:
        export_result = export_selection(
            self.crid,
            result,
            export_id=export_id,
            description=description,
            overwrite=self.overwrite,
        )
        urls = get_export_urls(self.crid, export_result.export_id)
        batch_result = create_batch(
            export_result.export_id,
            urls,
            project_id=self.sama_project_id,
            priority=self.priority,
        )
        sama_batch_id = batch_result.get("batch_id") or batch_result.get("id")
        if sama_batch_id is None:
            raise RuntimeError(
                f"Sama create_batch returned no batch_id: {batch_result!r}",
            )
        return CridSamaSinkResult(
            export_id=export_result.export_id,
            description=export_result.description,
            sama_batch_id=str(sama_batch_id),
            image_count=len(result.selected_ids),
        )
