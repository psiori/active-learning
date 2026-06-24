"""Tests for the composite CRID->Sama sink."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from active_learning.sinks.sama import CridSamaSink
from active_learning.core.types import SelectionResult


def test_crid_sama_sink_exports_then_submits_batch():
    result = SelectionResult(selected_ids=["a", "b"])
    with (
        patch(
            "active_learning.sinks.sama.export_selection",
            return_value=type(
                "ExportResult",
                (),
                {"export_id": "export-x", "description": "desc"},
            )(),
        ) as export_mock,
        patch(
            "active_learning.sinks.sama.get_export_urls",
            return_value=["https://example.invalid/a", "https://example.invalid/b"],
        ) as urls_mock,
        patch(
            "active_learning.sinks.sama.create_batch",
            return_value={"batch_id": 901, "creation_status": {}},
        ) as batch_mock,
    ):
        sink = CridSamaSink(object(), sama_project_id=123, priority=7, overwrite=True)
        sink_result = sink.submit(result, export_id="export-x", description="desc")

    export_mock.assert_called_once()
    urls_mock.assert_called_once()
    batch_mock.assert_called_once_with(
        "export-x",
        ["https://example.invalid/a", "https://example.invalid/b"],
        project_id=123,
        priority=7,
    )
    assert sink_result.export_id == "export-x"
    assert sink_result.description == "desc"
    assert sink_result.sama_batch_id == "901"
    assert sink_result.image_count == 2


def test_crid_sama_sink_does_not_call_sama_when_export_fails():
    result = SelectionResult(selected_ids=["a"])
    with (
        patch(
            "active_learning.sinks.sama.export_selection",
            side_effect=RuntimeError("export failed"),
        ),
        patch("active_learning.sinks.sama.create_batch") as batch_mock,
    ):
        sink = CridSamaSink(object(), sama_project_id=123)
        with pytest.raises(RuntimeError, match="export failed"):
            sink.submit(result, export_id="export-x", description="desc")
    batch_mock.assert_not_called()


def test_crid_sama_sink_propagates_sama_failure_after_export():
    result = SelectionResult(selected_ids=["a"])
    with (
        patch(
            "active_learning.sinks.sama.export_selection",
            return_value=type(
                "ExportResult",
                (),
                {"export_id": "export-x", "description": "desc"},
            )(),
        ),
        patch(
            "active_learning.sinks.sama.get_export_urls",
            return_value=["https://example.invalid/a"],
        ),
        patch(
            "active_learning.sinks.sama.create_batch",
            side_effect=RuntimeError("sama failed"),
        ),
    ):
        sink = CridSamaSink(object(), sama_project_id=123)
        with pytest.raises(RuntimeError, match="sama failed"):
            sink.submit(result, export_id="export-x", description="desc")
