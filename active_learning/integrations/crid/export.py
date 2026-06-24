"""CRID export helpers for active-learning scripts and sinks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import re

import pandas as pd

from active_learning.core.config import ConfigError, SeedConfig
from active_learning.core.sample_id import datetime_from_sample_id
from active_learning.core.types import SelectionResult


_MAX_AZURE_EXPORT_CONTAINER_NAME_LEN = 63


def export_segment(label: str) -> str:
    """Normalize free-form export labels into Azure/container-safe path segments."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(label).strip())
    return s.strip("_") or "x"


def selected_image_time_bounds(
    selected_ids: list[str],
) -> tuple[datetime, datetime] | None:
    """Return the min/max parsed timestamps for the selected sample IDs, if any."""
    timestamps = [
        dt
        for dt in (datetime_from_sample_id(sample_id) for sample_id in selected_ids)
        if dt is not None
    ]
    if not timestamps:
        return None
    return min(timestamps), max(timestamps)


def export_id_image_range_slug(bounds: tuple[datetime, datetime]) -> str:
    """Build the compact timestamp-range slug used inside generated export IDs."""
    t0, t1 = bounds
    return f"{t0.strftime('%y%m%d%H%M')}-{t1.strftime('%y%m%d%H%M')}"


def export_image_range_description(bounds: tuple[datetime, datetime]) -> str:
    """Build the human-readable time-range fragment for export descriptions."""
    t0, t1 = bounds
    return (
        f"Image range: {t0.strftime('%Y-%m-%d %H:%M:%S')} "
        f"to {t1.strftime('%Y-%m-%d %H:%M:%S')}"
    )


def _normalize_export_prefix(prefix: str | None) -> str:
    raw_prefix = (prefix or "").strip()
    if raw_prefix.startswith("export-"):
        raw_prefix = raw_prefix[len("export-") :].lstrip("-")
    return export_segment(raw_prefix) if raw_prefix else "x"


def _build_export_id(prefix: str, run_ts: str, range_slug: str) -> str:
    suffix = f"-{run_ts}-{range_slug}"
    export_id = f"export-{prefix}{suffix}"
    if len(export_id) <= _MAX_AZURE_EXPORT_CONTAINER_NAME_LEN:
        return export_id

    max_prefix_len = (
        _MAX_AZURE_EXPORT_CONTAINER_NAME_LEN - len("export-") - len(suffix)
    )
    if max_prefix_len < 1:
        raise ConfigError(
            f"export_id suffix is too long for Azure export containers "
            f"(max {_MAX_AZURE_EXPORT_CONTAINER_NAME_LEN} characters). "
            f"Suffix would be: {suffix!r}",
        )

    trimmed_prefix = prefix[:max_prefix_len].rstrip("-") or "x"
    export_id = f"export-{trimmed_prefix}{suffix}"
    if len(export_id) > _MAX_AZURE_EXPORT_CONTAINER_NAME_LEN:
        raise ConfigError(
            f"export_id is {len(export_id)} characters; Azure export containers allow at most "
            f"{_MAX_AZURE_EXPORT_CONTAINER_NAME_LEN}. Shorten export.prefix or narrow the query window. "
            f"export_id would be: {export_id!r}",
        )
    return export_id


def export_id_and_range_description(
    result: SelectionResult,
    cfg: SeedConfig,
) -> tuple[str, str]:
    """Generate the export container ID and matching range description.

    When sample IDs contain parseable timestamps, the export name is based on
    the actual selected-image time range. Otherwise it falls back to the query
    window. The generated ID is validated against Azure's 63-character container
    name limit, with compact fallbacks for window-based names and long prefixes.
    """
    run_ts = datetime.now().strftime("%y%m%d%H%M")
    bounds = selected_image_time_bounds(result.selected_ids)
    if bounds:
        range_slug = export_id_image_range_slug(bounds)
        range_desc = export_image_range_description(bounds)
    else:
        range_slug = (
            f"win-{export_segment(cfg.query.start)}-to-{export_segment(cfg.query.end)}"
        )
        range_desc = (
            f"Query window (could not parse image times from IDs): "
            f"{cfg.query.start} to {cfg.query.end}"
        )

    prefix = _normalize_export_prefix(cfg.export.prefix)
    export_id = f"export-{prefix}-{run_ts}-{range_slug}"
    if len(export_id) <= _MAX_AZURE_EXPORT_CONTAINER_NAME_LEN:
        return export_id, range_desc

    if bounds is None:
        compact_start = re.sub(r"[^0-9]", "", str(cfg.query.start))[:8]
        compact_end = re.sub(r"[^0-9]", "", str(cfg.query.end))[:8]
        compact_slug = f"w-{compact_start}-{compact_end}"
        compact_id = f"export-{prefix}-{run_ts}-{compact_slug}"
        if len(compact_id) <= _MAX_AZURE_EXPORT_CONTAINER_NAME_LEN:
            return compact_id, range_desc
        range_slug = compact_slug

    return _build_export_id(prefix, run_ts, range_slug), range_desc


@dataclass(frozen=True, slots=True)
class CridExportResult:
    """Minimal result payload returned after exporting a selection to CRID."""

    export_id: str
    description: str


def export_selection(
    crid,
    result: SelectionResult,
    *,
    export_id: str,
    description: str,
    overwrite: bool = False,
) -> CridExportResult:
    """Export the selected blob IDs as a CRID dataset and return its metadata."""
    from interface.model.models import Dataset

    account_url = crid.azure_client.config.account_url
    container = crid.config.azure.data_container
    sas_token = crid.azure_client.get_sas_token(container, expiry_in_days=7)
    image_url = crid.dataset_service._create_blob_url_lambda(
        account_url,
        container,
        sas_token,
    )
    image_url_no_sas = crid.dataset_service._create_blob_url_no_sas_lambda(
        account_url,
        container,
    )

    dataset = Dataset()
    dataset.name = export_id
    dataset.description = description
    dataset.blob_ids = list(result.selected_ids)
    dataset.dataframe = pd.DataFrame({"blob": result.selected_ids})
    dataset.dataframe["image_url_no_sas"] = dataset.dataframe["blob"].map(
        image_url_no_sas,
    )
    dataset.dataframe["image_url"] = dataset.dataframe["blob"].map(image_url)
    crid.export_dataset(
        dataset,
        export_id=export_id,
        description=description,
        overwrite=overwrite,
    )
    return CridExportResult(export_id=export_id, description=description)


def get_export_urls(
    crid,
    export_id: str,
) -> list[str]:
    """Return the anonymous URLs for a previously exported CRID dataset."""
    mapping = crid.export_service.get_mapping(export_id)
    if mapping is None or mapping.empty:
        raise RuntimeError(
            f"Export mapping for {export_id!r} is missing or empty in Azure. "
            "Wait for the CRID export to finish writing the mapping, or check the export id.",
        )
    return mapping["anon_URL"].tolist()


def build_export_description(
    result: SelectionResult,
    cfg,
    strategy: str,
) -> tuple[str, str]:
    """Build the export ID and human-readable description for a selection run."""
    export_id, range_desc = export_id_and_range_description(result, cfg)
    description = (
        f"Active learning: {len(result.selected_ids)} images selected with {strategy}. "
        f"{range_desc}"
    )
    if cfg.export.project:
        description = f"[{cfg.export.project}] {description}"
    if strategy in (
        "uncertainty_coreset",
        "uncertainty_topk",
        "uncertainty_topk_coreset",
    ):
        extras = [
            cfg.uncertainty_coreset.provider,
            f"aggregation={cfg.uncertainty_coreset.aggregation}",
        ]
        if strategy == "uncertainty_coreset":
            extras.append(f"alpha={cfg.uncertainty_coreset.alpha:.1f}")
        description += f" ({', '.join(extras)})"
    return export_id, description


def build_interactive_seed_payload(
    result: SelectionResult,
    cfg,
    strategy: str,
) -> dict:
    """Serialize a seed run into the payload consumed by the interactive workflow."""
    export_id, description = build_export_description(result, cfg, strategy)
    return {
        "version": 1,
        "config": {
            "models": {
                name: {k: v for k, v in asdict(model).items() if k != "name"}
                for name, model in cfg.models.items()
            },
            "query": asdict(cfg.query),
            "selection": asdict(cfg.selection),
            "coreset": asdict(cfg.coreset),
            "uncertainty_coreset": asdict(cfg.uncertainty_coreset),
            "alges": asdict(cfg.alges),
            "export": asdict(cfg.export),
        },
        "export": {
            "export_id": export_id,
            "description": description,
            "project": getattr(cfg.export, "project", None),
            "prefix": getattr(cfg.export, "prefix", None),
            "sama_project_id": getattr(cfg.export, "sama_project_id", None),
            "sama_priority": getattr(cfg.query, "sama_priority", 0),
        },
        "query": {
            "sensor": getattr(cfg.query, "sensor", None),
            "start": getattr(cfg.query, "start", None),
            "end": getattr(cfg.query, "end", None),
            "use_full_res_images": getattr(cfg.query, "use_full_res_images", False),
        },
        "selection": {
            "strategy": strategy,
            "n_select": len(result.selected_ids),
            "seed": getattr(cfg.selection, "seed", None),
        },
        "sample_ids": list(result.selected_ids),
    }
