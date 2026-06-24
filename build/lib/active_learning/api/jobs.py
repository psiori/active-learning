from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from collections.abc import Callable, Generator
from typing import Any

from active_learning.api.progress import JobCancelledError, JobState, ProgressReporter

logger = logging.getLogger(__name__)


_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        self._queues: dict[str, list[queue.Queue[dict[str, Any]]]] = {}
        self._cancel_events: dict[str, threading.Event] = {}

    def create_job(self, kind: str) -> JobState:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        state = JobState(
            job_id=job_id,
            kind=kind,
            state="queued",
            stage="queued",
            message="Job created",
        )
        with self._lock:
            self._jobs[job_id] = state
            self._queues[job_id] = []
        return state

    def get(self, job_id: str) -> JobState:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def update(self, job_id: str, *, event: str, **fields) -> None:
        with self._lock:
            state = self._jobs[job_id]
            for key, value in fields.items():
                setattr(state, key, value)
            payload = state.to_dict()
            queues = list(self._queues.get(job_id, []))
        self._broadcast(job_id, event, payload, queues)

    def add_skipped_stages(self, job_id: str, stage_ids: tuple[str, ...]) -> None:
        with self._lock:
            state = self._jobs[job_id]
            merged = list(state.skipped_stages)
            for sid in stage_ids:
                if sid not in merged:
                    merged.append(sid)
            state.skipped_stages = merged
            payload = state.to_dict()
            queues = list(self._queues.get(job_id, []))
        self._broadcast(job_id, "progress", payload, queues)

    def add_cached_stages(self, job_id: str, stage_ids: tuple[str, ...]) -> None:
        with self._lock:
            state = self._jobs[job_id]
            merged = list(state.cached_stages)
            for sid in stage_ids:
                if sid not in merged:
                    merged.append(sid)
            state.cached_stages = merged
            payload = state.to_dict()
            queues = list(self._queues.get(job_id, []))
        self._broadcast(job_id, "progress", payload, queues)

    def complete(self, job_id: str, *, result: dict[str, Any], message: str) -> None:
        self.update(
            job_id,
            event="result",
            state="completed",
            stage="done",
            message=message,
            completed=1,
            total=1,
            percent=100.0,
            result=result,
            error=None,
        )
        self._end(job_id)

    def fail(self, job_id: str, exc: Exception) -> None:
        self.update(
            job_id,
            event="error",
            state="failed",
            message=str(exc),
            error={"type": exc.__class__.__name__, "detail": str(exc)},
        )
        self._end(job_id)

    def cancel(self, job_id: str) -> None:
        self.update(job_id, event="status", state="cancelled", message="Job cancelled")
        self._end(job_id)

    def cancel_job(self, job_id: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            if self._jobs[job_id].state in _TERMINAL_STATES:
                return
        self._cancel_events[job_id].set()

    def start_job(
        self, kind: str, worker: Callable[[ProgressReporter], dict[str, Any]]
    ) -> JobState:
        state = self.create_job(kind)
        cancel_event = threading.Event()
        self._cancel_events[state.job_id] = cancel_event

        def run() -> None:
            reporter = ProgressReporter(self, state.job_id, cancel_event)
            try:
                result = worker(reporter)
                reporter.complete(result=result, message=f"{kind.title()} completed")
            except JobCancelledError:
                reporter.cancel()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Job %s (%s) failed: %s", state.job_id, kind, exc)
                reporter.fail(exc)

        thread = threading.Thread(target=run, name=state.job_id, daemon=True)
        thread.start()
        return state

    def event_stream(self, job_id: str) -> Generator[str, None, None]:
        q: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            self._queues[job_id].append(q)
            snapshot = self._jobs[job_id].to_dict()
        yield self._format_sse("snapshot", snapshot)
        try:
            while True:
                try:
                    item = q.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield self._format_sse(item["event"], item["data"])
                if item["event"] == "end":
                    break
        finally:
            with self._lock:
                queues = self._queues.get(job_id, [])
                if q in queues:
                    queues.remove(q)

    def _broadcast(
        self,
        job_id: str,
        event: str,
        data: dict[str, Any],
        queues: list[queue.Queue[dict[str, Any]]] | None = None,
    ) -> None:
        payload = {"event": event, "data": data}
        for q in queues or []:
            q.put(payload)

    def _end(self, job_id: str) -> None:
        with self._lock:
            queues = list(self._queues.get(job_id, []))
        self._broadcast(job_id, "end", {"job_id": job_id}, queues)

    @staticmethod
    def _format_sse(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"


job_manager = JobManager()
