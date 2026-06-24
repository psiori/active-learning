"""Sink for writing the interactive seed YAML handoff."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import os
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML as _YAML

from active_learning.core.types import SelectionResult


def load_interactive_seed_input(path: str | Path) -> dict:
    payload_path = Path(path)
    if payload_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"{payload_path} must be a .yaml or .yml handoff file")
    with open(payload_path, encoding="utf-8") as f:
        data = _YAML(typ="safe").load(f) or {}
    if "sample_ids" not in data:
        raise ValueError(f"{payload_path} does not contain sample_ids")
    return data


def write_interactive_seed_payload(
    result: SelectionResult,
    cfg,
    *,
    mosaic_path: str,
) -> str:
    from active_learning.integrations.crid.export import build_interactive_seed_payload

    payload = build_interactive_seed_payload(result, cfg, cfg.selection.strategy)
    payload["mosaic_path"] = mosaic_path
    return write_selection_payload(payload, mosaic_path)


def write_local_selection_payload(
    result: SelectionResult,
    cfg,
    *,
    mosaic_path: str,
    images_dir: str | Path,
) -> str:
    """Write a CRID-free YAML handoff for a local-image selection run."""
    payload = {
        "version": 1,
        "kind": "local_selection",
        "source": {
            "type": "local_directory",
            "images_dir": str(Path(images_dir).expanduser().resolve()),
        },
        "config": _config_payload(cfg),
        "mosaic_path": str(mosaic_path),
        "selection": {
            "strategy": cfg.selection.strategy,
            "n_select": len(result.selected_ids),
            "seed": getattr(cfg.selection, "seed", None),
        },
        "sample_ids": list(result.selected_ids),
    }
    return write_selection_payload(payload, mosaic_path)


def write_selection_payload(payload: dict[str, Any], mosaic_path: str | Path) -> str:
    output_path = os.path.splitext(str(mosaic_path))[0] + ".yaml"
    _yaml = _YAML(typ="safe")
    _yaml.default_flow_style = False
    with open(output_path, "w", encoding="utf-8") as f:
        _yaml.dump(payload, f)
    return output_path


def _config_payload(cfg) -> dict[str, Any]:
    return {
        "models": {
            name: {
                key: value
                for key, value in _to_plain(asdict(model)).items()
                if key != "name"
            }
            for name, model in cfg.models.items()
        },
        "query": _to_plain(asdict(cfg.query)),
        "selection": _to_plain(asdict(cfg.selection)),
        "coreset": _to_plain(asdict(cfg.coreset)),
        "uncertainty_coreset": _to_plain(asdict(cfg.uncertainty_coreset)),
        "alges": _to_plain(asdict(cfg.alges)),
        "export": _to_plain(asdict(cfg.export)),
    }


def _to_plain(value):
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value
