"""Configuration system for active learning scripts.

Provides YAML config loading, CLI override merging, and validated
dataclasses. Designed so any script can adopt --config support by
using the generic utilities (load_yaml, merge_cli, _UNSET) and
shared dataclasses (ModelDef, QueryConfig).

Priority: CLI args > YAML file > dataclass defaults.
"""

import logging
import os
import re
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlsplit

from ruamel.yaml import YAML as _YAML

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel for "not provided" (distinguishes from None or argparse defaults)
# ---------------------------------------------------------------------------


class _UnsetType:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<UNSET>"

    def __bool__(self):
        return False


_UNSET = _UnsetType()


class ConfigError(Exception):
    """Raised when config validation fails."""

    pass


def normalize_iso_datetime_string(raw: str) -> str:
    """Normalize user/YAML strings so :func:`datetime.fromisoformat` accepts them."""
    s = str(raw).strip()
    if s.endswith("Z") and "T" in s:
        return s[:-1] + "+00:00"
    return s


def parse_query_datetime(value: str) -> datetime:
    """Parse *value* as an ISO-8601 local or offset datetime (date-only allowed)."""
    try:
        return datetime.fromisoformat(normalize_iso_datetime_string(value))
    except ValueError as e:
        raise ConfigError(
            f"Invalid ISO datetime: {value!r}. Use e.g. 2026-04-15, "
            "2026-04-15T14:30:00, 2026-04-15 14:30:00, or 2026-04-15T12:00:00Z.",
        ) from e


def parse_boolish(value) -> bool:
    """Coerce config/CLI values to bool.

    argparse passes ``'false'`` and ``'False'`` as strings, which are truthy in
    Python if used as-is. YAML usually loads real bools. Accept common spellings
    and numeric 0/1.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "0", "false", "no", "off", "n", "f"):
            return False
        if s in ("1", "true", "yes", "on", "y", "t"):
            return True
        raise ConfigError(
            f"Invalid boolean: {value!r}. Use true/false, 1/0, or yes/no.",
        )
    return bool(value)


# ---------------------------------------------------------------------------
# Generic utilities (reusable by any script)
# ---------------------------------------------------------------------------


def load_yaml(path: str | None) -> dict:
    """Load a YAML file. Returns empty dict if path is None."""
    if path is None:
        return {}
    with open(path) as f:
        data = _YAML(typ="safe").load(f)
    return data or {}


def merge_cli(yaml_dict: dict, cli_args, mapping: dict[str, tuple[str, str]]) -> dict:
    """Overlay explicitly-provided CLI args onto a YAML dict.

    Only values that are not _UNSET are merged (i.e., only values the
    user actually typed on the command line).

    Args:
        yaml_dict: Parsed YAML config (will be deep-copied).
        cli_args: argparse.Namespace with _UNSET defaults.
        mapping: Maps argparse dest name -> (yaml_section, yaml_key).

    Returns:
        Merged dict with CLI overrides applied.
    """
    result = deepcopy(yaml_dict)

    for attr, target in mapping.items():
        val = getattr(cli_args, attr, _UNSET)
        if val is _UNSET:
            continue
        section, key = target
        result.setdefault(section, {})[key] = val

    return result


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into *base*. Returns a new dict.

    - Dicts are merged recursively (overlay keys win on conflict).
    - Non-dict values in overlay replace those in base.
    """
    result = deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _deep_merge(base: dict, overlay: dict) -> dict:
    return deep_merge(base, overlay)


# ---------------------------------------------------------------------------
# Shared dataclasses (usable by any script)
# ---------------------------------------------------------------------------

VALID_MODEL_TYPES = ("resnet50", "unet")
QUERY_CONFIG_KEYS = {
    "sensor",
    "sama_priority",
    "exclude_seeded",
    "exclude_al_excluded",
    "cache_root",
    "min_brightness",
    "max_brightness",
    "brightness_filter_enabled",
    "start",
    "end",
    "start_date",
    "end_date",
    "use_full_res_images",
    "min_milliseconds_between_images",
}


@dataclass
class ModelDef:
    name: str
    type: str
    path: str | None = None
    url: str | None = None
    image_size: tuple[int, int] = (224, 224)

    def __post_init__(self):
        if self.type not in VALID_MODEL_TYPES:
            raise ConfigError(
                f"Unknown model type '{self.type}' for model '{self.name}'. Valid: {', '.join(VALID_MODEL_TYPES)}",
            )
        if self.type == "unet" and not self.path and not self.url:
            raise ConfigError(
                f"Model '{self.name}' (type=unet) requires 'path' or 'url'",
            )
        if isinstance(self.image_size, list):
            self.image_size = tuple(self.image_size)


@dataclass
class QueryConfig:
    sensor: str | None = None
    sama_priority: int = 0
    exclude_seeded: bool = False
    exclude_al_excluded: bool = False
    cache_root: str = "/crid/jupyterhub/.al_feature_cache"
    min_brightness: float = 0.0
    max_brightness: float = 220.0
    brightness_filter_enabled: bool = True
    start: str = "2020-01-01"
    end: str = "2027-01-01"
    use_full_res_images: bool = False
    max_size: int = 1000000
    min_milliseconds_between_images: float = 0.0

    def __post_init__(self):
        lo, hi = -(2**31), 2**31 - 1
        if not isinstance(self.sama_priority, int):
            raise ConfigError(
                f"query.sama_priority must be an int, got {type(self.sama_priority).__name__}",
            )
        if not lo <= self.sama_priority <= hi:
            raise ConfigError(
                f"query.sama_priority must be between {lo} and {hi}, got {self.sama_priority}",
            )
        t0 = parse_query_datetime(self.start)
        t1 = parse_query_datetime(self.end)
        if t1 < t0:
            raise ConfigError(
                f"query.end must be >= query.start (got end={t1} before start={t0})",
            )
        if self.min_milliseconds_between_images < 0:
            raise ConfigError(
                "query.min_milliseconds_between_images must be >= 0, got "
                f"{self.min_milliseconds_between_images}",
            )


def brightness_filter_inactive(query: QueryConfig) -> bool:
    """True when brightness scoring / filtering must not run."""
    if not getattr(query, "brightness_filter_enabled", True):
        return True
    return query.min_brightness <= 0 and query.max_brightness >= 255


def _download_model(url: str, path: str) -> None:
    """Download a model file from a URL to a local path."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    logger.info("Downloading model to %s ...", path)
    urllib.request.urlretrieve(url, path)
    logger.info("Download complete (%d bytes)", os.path.getsize(path))


def normalize_model_paths(models: dict[str, "ModelDef"]) -> None:
    """Populate default local paths for models that only specify a URL."""
    for model in models.values():
        if model.path:
            continue
        if not model.url:
            continue
        filename = os.path.basename(urlsplit(model.url).path)
        if not filename:
            filename = f"{model.name}.zip"
        model.path = os.path.join(".", "data", "models", filename)


def ensure_models_downloaded(models: dict[str, "ModelDef"]) -> None:
    """Download any models that have a url but are missing locally.

    If a model has a url and no path, a default path is derived from the
    URL filename under ./data/models/.

    Skips download if the path already exists.
    """
    normalize_model_paths(models)
    for model in models.values():
        if not model.url:
            continue
        if os.path.exists(model.path):
            logger.info(
                "Model %r already exists at %s, skipping download",
                model.name,
                model.path,
            )
            continue
        _download_model(model.url, model.path)


# ---------------------------------------------------------------------------
# Project resolution
# ---------------------------------------------------------------------------


def resolve_project(yaml_dict: dict, project_name: str | None) -> dict:
    """Overlay a named project's settings onto the config dict.

    Each key in the project block is deep-merged into the matching
    top-level section (query, alges, export, models, ...).

    Applied after YAML loading but before CLI merging, so CLI args
    still take priority.
    """
    if project_name is None:
        return yaml_dict

    projects = yaml_dict.get("projects", {})
    if project_name not in projects:
        available = ", ".join(projects.keys()) or "(none)"
        raise ConfigError(f"Unknown project '{project_name}'. Available: {available}")

    result = deepcopy(yaml_dict)
    project = projects[project_name]

    for key, value in project.items():
        if isinstance(value, dict):
            result[key] = _deep_merge(result.get(key, {}), value)
        else:
            result[key] = deepcopy(value)

    result.setdefault("export", {})["project"] = project_name

    return result


def parse_models(yaml_dict: dict) -> dict[str, ModelDef]:
    """Parse models section. Always includes a resnet50 entry."""
    models = {}
    for name, mdef in yaml_dict.get("models", {}).items():
        mdef = dict(mdef)  # copy
        models[name] = ModelDef(name=name, **mdef)
    # Ensure resnet50 always exists
    if "resnet50" not in models:
        models["resnet50"] = ModelDef(
            name="resnet50",
            type="resnet50",
            image_size=(224, 224),
        )
    return models


def parse_query(yaml_dict: dict) -> QueryConfig:
    """Parse query section with defaults.

    Accepts legacy YAML keys ``start_date`` / ``end_date`` as aliases for ``start`` / ``end``.
    """
    if "query" in yaml_dict and isinstance(yaml_dict.get("query"), dict):
        q = dict(yaml_dict.get("query") or {})
    else:
        q = {
            key: value
            for key, value in dict(yaml_dict or {}).items()
            if key in QUERY_CONFIG_KEYS
        }
    if "start" in q:
        q.pop("start_date", None)
    elif "start_date" in q:
        q["start"] = q.pop("start_date")
    if "end" in q:
        q.pop("end_date", None)
    elif "end_date" in q:
        q["end"] = q.pop("end_date")
    if "exclude_seeded" in q:
        q["exclude_seeded"] = parse_boolish(q["exclude_seeded"])
    if "exclude_al_excluded" in q:
        q["exclude_al_excluded"] = parse_boolish(q["exclude_al_excluded"])
    if "use_full_res_images" in q:
        q["use_full_res_images"] = parse_boolish(q["use_full_res_images"])
    if "brightness_filter_enabled" in q:
        q["brightness_filter_enabled"] = parse_boolish(q["brightness_filter_enabled"])
    # YAML `null` would otherwise become Python None and break numeric comparisons
    # (and the Vue range slider). Omit so QueryConfig dataclass defaults apply.
    if q.get("min_brightness") is None:
        q.pop("min_brightness", None)
    if q.get("max_brightness") is None:
        q.pop("max_brightness", None)
    return QueryConfig(**q)


# ---------------------------------------------------------------------------
# Seed-specific dataclasses
# ---------------------------------------------------------------------------

VALID_STRATEGIES = (
    "coreset",
    "uncertainty_coreset",
    "uncertainty_topk",
    "uncertainty_topk_coreset",
    "alges",
    "alges_coreset",
)
VALID_PROVIDERS = ("mc_dropout", "entropy", "bald")
VALID_UNCERTAINTY_AGGREGATIONS = ("mean", "topk_mean", "max")
VALID_ALGES_METHODS = ("image", "semantic")


@dataclass
class SelectionConfig:
    strategy: str = "alges"
    n_select: int = 200
    seed: int = 42

    def __post_init__(self):
        if self.strategy not in VALID_STRATEGIES:
            raise ConfigError(
                f"Unknown strategy '{self.strategy}'. Valid: {', '.join(VALID_STRATEGIES)}",
            )
        if self.n_select <= 0:
            raise ConfigError(f"n_select must be positive, got {self.n_select}")


@dataclass
class CoresetConfig:
    feature_model: str = "resnet50"


@dataclass
class UncertaintyCoresetConfig:
    feature_model: str = "resnet50"
    uncertainty_model: str = "geiles_unet_250912"
    alpha: float = 0.5
    provider: str = "mc_dropout"
    mc_iterations: int = 5
    batch_size: int = 32
    aggregation: str = "topk_mean"
    topk_fraction: float = 0.10
    candidate_multiplier: int = 4
    target_classes: list[int] | None = None

    def __post_init__(self):
        if self.target_classes is None:
            self.target_classes = []
        else:
            self.target_classes = [int(class_id) for class_id in self.target_classes]
        if not 0.0 <= self.alpha <= 1.0:
            raise ConfigError(f"alpha must be in [0, 1], got {self.alpha:.2f}")
        if self.provider not in VALID_PROVIDERS:
            raise ConfigError(
                f"Unknown provider '{self.provider}'. Valid: {', '.join(VALID_PROVIDERS)}",
            )
        if self.mc_iterations <= 0:
            raise ConfigError("mc_iterations must be positive")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be positive")
        if self.aggregation not in VALID_UNCERTAINTY_AGGREGATIONS:
            raise ConfigError(
                "Unknown aggregation "
                f"'{self.aggregation}'. Valid: {', '.join(VALID_UNCERTAINTY_AGGREGATIONS)}",
            )
        if not 0.0 < self.topk_fraction <= 1.0:
            raise ConfigError(
                f"topk_fraction must be in (0, 1], got {self.topk_fraction}",
            )
        if self.candidate_multiplier <= 0:
            raise ConfigError("candidate_multiplier must be positive")


@dataclass
class AlgesConfig:
    model: str = "geiles_unet_250912"
    method: str = "semantic"
    batch_size: int = 32

    def __post_init__(self):
        if self.method not in VALID_ALGES_METHODS:
            raise ConfigError(
                f"Unknown ALGES method '{self.method}'. Valid: {', '.join(VALID_ALGES_METHODS)}",
            )
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be positive")


@dataclass
class ExportConfig:
    sama_project_id: int | None = None
    prefix: str | None = None
    project: str | None = None
    mosaic_path: str = "/crid/jupyterhub/robin/mosaic_seed.jpg"
    seed: bool = False
    overlay: bool = False


@dataclass
class SeedConfig:
    models: dict[str, ModelDef]
    query: QueryConfig
    selection: SelectionConfig
    coreset: CoresetConfig
    uncertainty_coreset: UncertaintyCoresetConfig
    alges: AlgesConfig
    export: ExportConfig


def _require_model(models: dict[str, ModelDef], name: str, field_name: str):
    if name not in models:
        raise ConfigError(
            f"Model '{name}' referenced by {field_name} is not defined in the models section. "
            f"Available: {', '.join(models.keys())}",
        )


def build_seed_config(
    yaml_dict: dict,
    *,
    ensure_model_downloads: bool = True,
    require_sensor: bool = True,
) -> SeedConfig:
    """Build and cross-validate a full SeedConfig from a merged dict.

    Validates:
    - Referenced model names exist
    - UNet-requiring fields reference unet-type models
    - export.sama_project_id present when exclude_seeded=True
    """
    models = parse_models(yaml_dict)
    normalize_model_paths(models)
    if ensure_model_downloads:
        ensure_models_downloaded(models)
    query = parse_query(yaml_dict)
    selection = SelectionConfig(**yaml_dict.get("selection", {}))
    coreset = CoresetConfig(**yaml_dict.get("coreset", {}))
    uc = UncertaintyCoresetConfig(**yaml_dict.get("uncertainty_coreset", {}))
    alges = AlgesConfig(**yaml_dict.get("alges", {}))
    export = ExportConfig(**yaml_dict.get("export", {}))

    # Cross-validate: model references
    strategy = selection.strategy
    if strategy == "coreset":
        _require_model(models, coreset.feature_model, "coreset.feature_model")
    elif strategy in (
        "uncertainty_coreset",
        "uncertainty_topk",
        "uncertainty_topk_coreset",
    ):
        _require_model(models, uc.feature_model, "uncertainty_coreset.feature_model")
        _require_model(
            models,
            uc.uncertainty_model,
            "uncertainty_coreset.uncertainty_model",
        )
        if models[uc.uncertainty_model].type != "unet":
            raise ConfigError(
                f"uncertainty_coreset.uncertainty_model '{uc.uncertainty_model}' must be type "
                f"'unet', got '{models[uc.uncertainty_model].type}'",
            )
    elif strategy in ("alges", "alges_coreset"):
        _require_model(models, alges.model, "alges.model")
        if models[alges.model].type != "unet":
            raise ConfigError(
                f"alges.model '{alges.model}' must be type 'unet', got '{models[alges.model].type}'",
            )
        if strategy == "alges_coreset":
            _require_model(models, coreset.feature_model, "coreset.feature_model")

    # Cross-validate: sensor required for CRID-backed runs.
    if require_sensor and query.sensor is None:
        raise ConfigError(
            "query.sensor is required. Set it via --project, --sensor, or in the YAML.",
        )

    # Cross-validate: export prefix required when exporting/seeding
    if export.seed and export.prefix is None:
        raise ConfigError(
            "export.prefix is required when exporting. Set it via --project, "
            "--export-prefix, or in the YAML.",
        )

    # Cross-validate: sama
    if query.exclude_seeded and export.sama_project_id is None:
        if not os.environ.get("SAMA_PROJECT_ID"):
            raise ConfigError(
                "exclude_seeded=true requires export.sama_project_id in config "
                "or SAMA_PROJECT_ID env var. Set exclude_seeded: false or provide it.",
            )

    return SeedConfig(
        models=models,
        query=query,
        selection=selection,
        coreset=coreset,
        uncertainty_coreset=uc,
        alges=alges,
        export=export,
    )


def handle_model_path_override(yaml_dict: dict, cli_args) -> dict:
    """Handle --model-path / --model-name convenience flags.

    Creates or updates a model entry in the models section when
    --model-path is provided on the CLI. Also routes the model name
    to the active strategy's model field.
    """
    model_path = getattr(cli_args, "model_path", _UNSET)
    if model_path is _UNSET:
        return yaml_dict

    result = deepcopy(yaml_dict)

    model_name = getattr(cli_args, "model_name", _UNSET)
    if model_name is _UNSET:
        # Derive from filename: geiles_unet_250912_0000000_custom_fold1.zip -> geiles_unet_250912
        base = os.path.splitext(os.path.basename(model_path))[0]
        model_name = re.split(r"_\d{7}_", base)[0] or base

    result.setdefault("models", {})[model_name] = {
        "type": "unet",
        "path": model_path,
        "image_size": [320, 240],
    }

    # Route to active strategy
    strategy = result.get("selection", {}).get("strategy", "alges")
    if strategy in (
        "uncertainty_coreset",
        "uncertainty_topk",
        "uncertainty_topk_coreset",
    ):
        result.setdefault("uncertainty_coreset", {})["uncertainty_model"] = model_name
    elif strategy == "alges":
        result.setdefault("alges", {})["model"] = model_name

    return result
