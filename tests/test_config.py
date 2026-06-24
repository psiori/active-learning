"""Tests for active-learning config and export helpers."""

from __future__ import annotations

import argparse

import pytest

from active_learning.cli.common import SEED_CLI_MAP
from active_learning.core.config import (
    _UNSET,
    ConfigError,
    SelectionConfig,
    UncertaintyCoresetConfig,
    build_seed_config,
    merge_cli,
    parse_boolish,
)
from active_learning.integrations.crid.export import (
    _MAX_AZURE_EXPORT_CONTAINER_NAME_LEN,
    build_export_description,
    export_id_and_range_description,
)
from active_learning.core.types import SelectionResult


def test_parse_boolish_cli_false_string():
    assert parse_boolish("False") is False
    assert parse_boolish("true") is True


def test_exclude_seeded_string_false_from_merge_cli():
    """argparse gives 'False' as str; it must not stay truthy."""
    merged = merge_cli(
        {
            "models": {"u": {"type": "unet", "path": "/tmp/m.zip"}},
            "selection": {"strategy": "alges"},
            "alges": {"model": "u"},
            "query": {"sensor": "CAM_TROLLEY", "exclude_seeded": True},
        },
        argparse.Namespace(exclude_seeded="False"),
        SEED_CLI_MAP,
    )
    assert merged["query"]["exclude_seeded"] == "False"
    cfg = build_seed_config(merged, ensure_model_downloads=False)
    assert cfg.query.exclude_seeded is False


def test_merge_cli_ignores_unset_values():
    result = merge_cli(
        {"selection": {"n_select": 10}},
        argparse.Namespace(n_select=_UNSET, strategy=_UNSET),
        SEED_CLI_MAP,
    )
    assert result["selection"]["n_select"] == 10


def test_merge_cli_min_milliseconds_between_images():
    result = merge_cli(
        {},
        argparse.Namespace(min_milliseconds_between_images=120_000.0),
        SEED_CLI_MAP,
    )
    assert result["query"]["min_milliseconds_between_images"] == 120_000.0


def test_selection_config_accepts_current_strategies():
    assert SelectionConfig(strategy="uncertainty_topk").strategy == "uncertainty_topk"
    assert SelectionConfig(strategy="alges_coreset").strategy == "alges_coreset"


def test_query_config_rejects_negative_min_milliseconds_between_images():
    with pytest.raises(ConfigError, match="min_milliseconds_between_images"):
        build_seed_config(
            {
                "models": {"u": {"type": "unet", "path": "/tmp/model.zip"}},
                "selection": {"strategy": "alges"},
                "alges": {"model": "u"},
                "query": {
                    "sensor": "CAM_TROLLEY",
                    "min_milliseconds_between_images": -1.0,
                },
            },
            ensure_model_downloads=False,
        )


def test_uncertainty_config_accepts_bald():
    assert UncertaintyCoresetConfig(provider="bald").provider == "bald"


def test_build_seed_config_requires_sensor():
    cfg = {
        "models": {"u": {"type": "unet", "path": "/tmp/model.zip"}},
        "selection": {"strategy": "alges"},
        "alges": {"model": "u"},
    }
    try:
        build_seed_config(cfg, ensure_model_downloads=False)
    except ConfigError as exc:
        assert "query.sensor" in str(exc)
    else:
        raise AssertionError("Expected ConfigError")


def test_build_seed_config_can_skip_sensor_for_local_runs():
    cfg = build_seed_config(
        {
            "selection": {"strategy": "coreset"},
            "query": {"cache_root": "/tmp/cache"},
        },
        ensure_model_downloads=False,
        require_sensor=False,
    )
    assert cfg.query.sensor is None
    assert cfg.selection.strategy == "coreset"


def test_build_export_description_uses_selection_result():
    cfg = argparse.Namespace(
        selection=argparse.Namespace(n_select=2),
        export=argparse.Namespace(project="proj", prefix="pref"),
        query=argparse.Namespace(start="2026-04-01", end="2026-04-02"),
        uncertainty_coreset=argparse.Namespace(
            alpha=0.5,
            provider="bald",
            aggregation="topk_mean",
        ),
    )
    result = SelectionResult(
        selected_ids=["origin/images/cam/2026-04-01T10_00_00_000Z.png"],
    )
    export_id, description = build_export_description(
        result,
        cfg,
        "uncertainty_coreset",
    )
    assert export_id.startswith("export-")
    assert "[proj]" in description
    assert "alpha=0.5" in description


def test_export_id_truncates_long_prefix_for_image_bounds():
    cfg = argparse.Namespace(
        export=argparse.Namespace(prefix="al-aespjetway-cabin-wide"),
        query=argparse.Namespace(start="2020-01-01", end="2027-01-01"),
    )
    result = SelectionResult(
        selected_ids=[
            "aespjetway/images/cam_cabin_wide/2025/08/07/2025-08-07T04_39_00_000Z.png",
            "aespjetway/images/cam_cabin_wide/2025/08/11/2025-08-11T05_48_00_000Z.png",
        ],
    )
    export_id, _ = export_id_and_range_description(result, cfg)
    assert len(export_id) <= _MAX_AZURE_EXPORT_CONTAINER_NAME_LEN
    assert export_id.startswith("export-")
    assert "2508070439" in export_id
    assert "2508110548" in export_id
