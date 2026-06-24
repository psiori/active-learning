from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any


class JobCancelledError(Exception):
    pass


@dataclass(slots=True)
class JobState:
    job_id: str
    kind: str
    state: str
    stage: str | None = None
    message: str | None = None
    completed: int | None = None
    total: int | None = None
    percent: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    skipped_stages: list[str] = field(default_factory=list)
    cached_stages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgressReporter:
    def __init__(self, manager, job_id: str, cancel_event: threading.Event):
        self.manager = manager
        self.job_id = job_id
        self._cancel_event = cancel_event

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise JobCancelledError("Job was cancelled")

    def cancel(self) -> None:
        self.manager.cancel(self.job_id)

    def skip_stages(self, *stage_ids: str) -> None:
        self.manager.add_skipped_stages(self.job_id, stage_ids)

    def cache_stages(self, *stage_ids: str) -> None:
        self.manager.add_cached_stages(self.job_id, stage_ids)

    def status(self, stage: str, message: str) -> None:
        self.manager.update(
            self.job_id,
            event="status",
            state="running",
            stage=stage,
            message=message,
            completed=None,
            total=None,
            percent=None,
        )

    def progress(
        self,
        stage: str,
        completed: int,
        total: int,
        message: str | None = None,
    ) -> None:
        percent = 100.0 if total <= 0 else (completed / total) * 100.0
        self.manager.update(
            self.job_id,
            event="progress",
            state="running",
            stage=stage,
            message=message,
            completed=completed,
            total=total,
            percent=percent,
        )

    def complete(self, result: dict[str, Any], message: str) -> None:
        self.manager.complete(self.job_id, result=result, message=message)

    def fail(self, exc: Exception) -> None:
        self.manager.fail(self.job_id, exc)
