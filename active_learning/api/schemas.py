from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectSummary(BaseModel):
    project_name: str
    query_sensor: str | None = None
    sama_project_id: int | None = None
    export_prefix: str | None = None
    export_project: str | None = None
    selection_strategy: str


class QuerySettings(BaseModel):
    cache_root: str
    sama_priority: int
    exclude_seeded: bool
    exclude_al_excluded: bool
    min_brightness: float
    max_brightness: float
    brightness_filter_enabled: bool = True
    start: str
    end: str
    use_full_res_images: bool
    min_milliseconds_between_images: float


class SelectionSettings(BaseModel):
    strategy: str
    available_strategies: list[str]
    n_select: int
    seed: int


class ModelOption(BaseModel):
    name: str
    type: str
    path: str | None = None
    url: str | None = None
    image_size: tuple[int, int] = (224, 224)


class CoresetSettings(BaseModel):
    feature_model: str


class UncertaintySettings(BaseModel):
    feature_model: str
    uncertainty_model: str
    alpha: float
    provider: str
    mc_iterations: int
    batch_size: int
    aggregation: str
    topk_fraction: float
    candidate_multiplier: int


class AlgesSettings(BaseModel):
    model: str
    method: str
    batch_size: int


class SaveQuerySettings(BaseModel):
    sama_priority: int | None = None
    exclude_seeded: bool | None = None
    exclude_al_excluded: bool | None = None
    min_brightness: float | None = None
    max_brightness: float | None = None
    brightness_filter_enabled: bool | None = None
    start: str | None = None
    end: str | None = None
    use_full_res_images: bool | None = None
    min_milliseconds_between_images: float | None = None


class SaveSelectionSettings(BaseModel):
    strategy: str | None = None
    n_select: int | None = None


class SaveCoresetSettings(BaseModel):
    feature_model: str | None = None


class SaveUncertaintySettings(BaseModel):
    uncertainty_model: str | None = None
    model_url: str | None = None
    alpha: float | None = None
    provider: str | None = None
    mc_iterations: int | None = None
    batch_size: int | None = None
    aggregation: str | None = None
    topk_fraction: float | None = None
    candidate_multiplier: int | None = None


class SaveAlgesSettings(BaseModel):
    model: str | None = None
    model_url: str | None = None
    method: str | None = None
    batch_size: int | None = None


class SaveModelSettings(BaseModel):
    url: str | None = None


class SaveProjectConfigRequest(BaseModel):
    query: SaveQuerySettings | None = None
    selection: SaveSelectionSettings | None = None
    coreset: SaveCoresetSettings | None = None
    uncertainty_coreset: SaveUncertaintySettings | None = None
    alges: SaveAlgesSettings | None = None
    models: dict[str, SaveModelSettings] | None = None


class ProjectConfigResponse(BaseModel):
    project: ProjectSummary
    query: QuerySettings
    selection: SelectionSettings
    models: list[ModelOption]
    coreset: CoresetSettings
    uncertainty_coreset: UncertaintySettings
    alges: AlgesSettings


class ConfigResponse(BaseModel):
    config_path: str
    projects: list[ProjectSummary]


class QueryPreviewRequest(BaseModel):
    project_name: str
    query_result_token: str | None = None
    strategy: str | None = None
    n_select: int | None = None
    seed: int | None = None
    min_milliseconds_between_images: float | None = None
    feature_model: str | None = None
    uncertainty_model: str | None = None
    uncertainty_model_url: str | None = None
    alpha: float | None = None
    provider: str | None = None
    mc_iterations: int | None = None
    batch_size: int | None = None
    aggregation: str | None = None
    topk_fraction: float | None = None
    candidate_multiplier: int | None = None
    alges_model: str | None = None
    alges_model_url: str | None = None
    method: str | None = None
    alges_batch_size: int | None = None
    exclude_seeded: bool | None = None
    exclude_al_excluded: bool | None = None
    min_brightness: float | None = None
    max_brightness: float | None = None
    brightness_filter_enabled: bool | None = None
    start: str | None = None
    end: str | None = None
    use_full_res_images: bool | None = None
    sample_size: int = Field(default=60, ge=1, le=500)


class PreviewItem(BaseModel):
    sample_id: str
    thumbnail_url: str
    mask_url: str | None = None
    excluded: bool = False


class QueryPreviewResponse(BaseModel):
    result_kind: str
    project: ProjectSummary
    query: QuerySettings
    selection: SelectionSettings
    models: list[ModelOption]
    coreset: CoresetSettings
    uncertainty_coreset: UncertaintySettings
    alges: AlgesSettings
    all_ids_count: int
    labeled_ids_count: int
    seeded_ids_count: int = 0
    pool_ids_count: int
    brightness_filtered_ids_count: int
    preview_sample_ids: list[str]
    preview_items: list[PreviewItem]
    query_result_token: str | None = None
    query_cache_hit: bool = False
    selected_ids_count: int | None = None
    selected_ids: list[str] | None = None
    overlay_available: bool = False


class ExportSettings(BaseModel):
    sama_project_id: int | None = None
    sama_priority: int = 0


class ExportContext(BaseModel):
    project: ProjectSummary
    query: QuerySettings
    selection: SelectionSettings
    models: list[ModelOption]
    uncertainty_coreset: UncertaintySettings
    alges: AlgesSettings
    export: ExportSettings


class OverlayMosaicRequest(BaseModel):
    export_context: ExportContext
    selected_ids: list[str] = Field(default_factory=list)


class SeedExportRequest(BaseModel):
    export_context: ExportContext
    selected_ids: list[str] = Field(default_factory=list)


class ExportSeedResponse(BaseModel):
    export_id: str
    description: str
    sama_batch_id: str | None = None
    image_count: int


class ExclusionTagsRequest(BaseModel):
    selected_ids: list[str]
    excluded_ids: list[str]
    project_name: str | None = None


class ExclusionTagsResponse(BaseModel):
    updated_count: int


class JobCreateRequest(QueryPreviewRequest):
    kind: str


class JobCreateResponse(BaseModel):
    job_id: str
    kind: str
    state: str


class JobSnapshotResponse(BaseModel):
    job_id: str
    kind: str
    state: str
    stage: str | None = None
    message: str | None = None
    completed: int | None = None
    total: int | None = None
    percent: float | None = None
    result: dict | None = None
    error: dict | None = None
    skipped_stages: list[str] = Field(default_factory=list)
    cached_stages: list[str] = Field(default_factory=list)
