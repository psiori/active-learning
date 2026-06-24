from __future__ import annotations

import os
from pathlib import Path

from ruamel.yaml import YAML

from active_learning.api.schemas import (
    AlgesSettings,
    CoresetSettings,
    ConfigResponse,
    ModelOption,
    ProjectConfigResponse,
    ProjectSummary,
    QuerySettings,
    SaveProjectConfigRequest,
    SelectionSettings,
    UncertaintySettings,
)
from active_learning.core.config import (
    VALID_STRATEGIES,
    build_seed_config,
    load_yaml,
    resolve_project,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "al_config.yaml"


def get_default_config_path() -> Path:
    return DEFAULT_CONFIG_PATH


def get_active_crid_origin() -> str | None:
    for env_var in ("ORIGIN", "ENVIRONMENT"):
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def get_project_crid_name(raw_config: dict, project_name: str) -> str:
    projects = raw_config.get("projects") or {}
    project = projects.get(project_name) or {}
    return str(project.get("crid_origin") or project_name)


def project_is_visible(raw_config: dict, project_name: str) -> bool:
    active_origin = get_active_crid_origin()
    if not active_origin:
        return True
    return get_project_crid_name(raw_config, project_name) == active_origin


def build_project_summary(config_path: str | Path, project_name: str) -> ProjectSummary:
    raw = load_yaml(str(config_path))
    resolved = resolve_project(raw, project_name)
    cfg = build_seed_config(resolved, ensure_model_downloads=False)
    return ProjectSummary(
        project_name=project_name,
        query_sensor=cfg.query.sensor,
        sama_project_id=cfg.export.sama_project_id,
        export_prefix=cfg.export.prefix,
        export_project=cfg.export.project,
        selection_strategy=cfg.selection.strategy,
    )


def build_model_options(cfg) -> list[ModelOption]:
    return [
        ModelOption(
            name=model.name,
            type=model.type,
            path=model.path,
            url=model.url,
            image_size=tuple(getattr(model, "image_size", (224, 224))),
        )
        for model in cfg.models.values()
    ]


def list_projects(config_path: str | Path | None = None) -> ConfigResponse:
    path = Path(config_path or get_default_config_path())
    raw = load_yaml(str(path))
    projects = sorted(
        project_name
        for project_name in (raw.get("projects") or {}).keys()
        if project_is_visible(raw, project_name)
    )
    return ConfigResponse(
        config_path=str(path),
        projects=[
            build_project_summary(path, project_name) for project_name in projects
        ],
    )


def save_project_config(
    project_name: str,
    data: SaveProjectConfigRequest,
    config_path: str | Path | None = None,
) -> None:
    path = Path(config_path or get_default_config_path())
    _yaml = YAML()
    _yaml.preserve_quotes = True
    _yaml.width = 4096
    with open(path) as f:
        raw = _yaml.load(f)

    projects = raw.setdefault("projects", {})
    if project_name not in projects:
        raise ValueError(f"Project '{project_name}' not found in config.")
    project_block = projects[project_name]

    section_map = {
        "query": data.query,
        "selection": data.selection,
        "coreset": data.coreset,
        "uncertainty_coreset": data.uncertainty_coreset,
        "alges": data.alges,
    }
    for key, section in section_map.items():
        if section is not None:
            updates = section.model_dump(exclude_none=True, exclude={"model_url"})
            if updates:
                if key not in project_block or not isinstance(project_block[key], dict):
                    project_block[key] = {}
                project_block[key].update(updates)

    # Update model URLs in the top-level models dict.
    if "models" not in raw or not isinstance(raw["models"], dict):
        raw["models"] = {}
    top_models = raw["models"]
    if data.models:
        for name, settings in data.models.items():
            if not name:
                continue
            updates = settings.model_dump(exclude_none=True)
            if not updates:
                continue
            if name not in top_models or not isinstance(top_models[name], dict):
                top_models[name] = {}
            top_models[name].update(updates)

    # Backward compatibility for callers that still send nested model_url fields.
    if data.uncertainty_coreset and data.uncertainty_coreset.model_url is not None:
        name = data.uncertainty_coreset.uncertainty_model
        if name and name in top_models:
            top_models[name]["url"] = data.uncertainty_coreset.model_url
    if data.alges and data.alges.model_url is not None:
        name = data.alges.model
        if name and name in top_models:
            top_models[name]["url"] = data.alges.model_url

    with open(path, "w") as f:
        _yaml.dump(raw, f)


def get_project_config(
    project_name: str,
    config_path: str | Path | None = None,
) -> ProjectConfigResponse:
    path = Path(config_path or get_default_config_path())
    raw = load_yaml(str(path))
    if not project_is_visible(raw, project_name):
        active_origin = get_active_crid_origin()
        raise ValueError(
            f"Project '{project_name}' is not available for CRID origin '{active_origin}'."
        )
    summary = build_project_summary(path, project_name)
    resolved = resolve_project(raw, project_name)
    cfg = build_seed_config(resolved, ensure_model_downloads=False)
    return ProjectConfigResponse(
        project=summary,
        query=QuerySettings(
            cache_root=str(cfg.query.cache_root),
            sama_priority=cfg.query.sama_priority,
            exclude_seeded=cfg.query.exclude_seeded,
            exclude_al_excluded=cfg.query.exclude_al_excluded,
            min_brightness=cfg.query.min_brightness,
            max_brightness=cfg.query.max_brightness,
            brightness_filter_enabled=cfg.query.brightness_filter_enabled,
            start=cfg.query.start,
            end=cfg.query.end,
            use_full_res_images=cfg.query.use_full_res_images,
            min_milliseconds_between_images=cfg.query.min_milliseconds_between_images,
        ),
        selection=SelectionSettings(
            strategy=cfg.selection.strategy,
            available_strategies=list(VALID_STRATEGIES),
            n_select=cfg.selection.n_select,
            seed=cfg.selection.seed,
        ),
        models=build_model_options(cfg),
        coreset=CoresetSettings(
            feature_model=cfg.coreset.feature_model,
        ),
        uncertainty_coreset=UncertaintySettings(
            feature_model=cfg.uncertainty_coreset.feature_model,
            uncertainty_model=cfg.uncertainty_coreset.uncertainty_model,
            alpha=cfg.uncertainty_coreset.alpha,
            provider=cfg.uncertainty_coreset.provider,
            mc_iterations=cfg.uncertainty_coreset.mc_iterations,
            batch_size=cfg.uncertainty_coreset.batch_size,
            aggregation=cfg.uncertainty_coreset.aggregation,
            topk_fraction=cfg.uncertainty_coreset.topk_fraction,
            candidate_multiplier=cfg.uncertainty_coreset.candidate_multiplier,
        ),
        alges=AlgesSettings(
            model=cfg.alges.model,
            method=cfg.alges.method,
            batch_size=cfg.alges.batch_size,
        ),
    )
