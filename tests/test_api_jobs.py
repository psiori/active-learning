from __future__ import annotations

import time

from active_learning.api.jobs import JobManager


def test_job_manager_runs_worker_and_records_result():
    manager = JobManager()

    def worker(reporter):
        reporter.status("query_pool", "Querying")
        reporter.progress("query_pool", 1, 1, "Done")
        return {"result_kind": "query", "preview_items": []}

    job = manager.start_job("query", worker)

    deadline = time.time() + 2.0
    while time.time() < deadline:
        state = manager.get(job.job_id)
        if state.state == "completed":
            break
        time.sleep(0.01)

    state = manager.get(job.job_id)
    assert state.state == "completed"
    assert state.stage == "done"
    assert state.result["result_kind"] == "query"


def test_job_manager_add_skipped_stages_merges():
    manager = JobManager()

    def worker(reporter):
        reporter.skip_stages("fetch_seeded", "fetch_seeded")
        reporter.status("query_pool", "Querying")
        return {"result_kind": "query", "preview_items": []}

    job = manager.start_job("query", worker)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        state = manager.get(job.job_id)
        if state.state == "completed":
            break
        time.sleep(0.01)

    state = manager.get(job.job_id)
    assert state.skipped_stages == ["fetch_seeded"]
