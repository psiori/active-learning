from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

from active_learning.api.schemas import (
    AlgesSettings,
    ExportContext,
    ExportSettings,
    ModelOption,
    ProjectSummary,
    QuerySettings,
    SelectionSettings,
    UncertaintySettings,
)
from active_learning.api.query_service import (
    build_overlay_mosaic_download,
    apply_brightness_filter,
    run_query_preview,
    run_strategy_preview,
    seed_strategy_selection,
)
from active_learning.api.schemas import QueryPreviewRequest


def make_export_context(*, strategy="alges", sama_project_id=12, sama_priority=7):
    return ExportContext(
        project=ProjectSummary(
            project_name="demo",
            query_sensor="CAM_TROLLEY",
            sama_project_id=sama_project_id,
            export_prefix="pref",
            selection_strategy=strategy,
        ),
        query=QuerySettings(
            cache_root="/tmp/cache",
            sama_priority=sama_priority,
            exclude_seeded=False,
            exclude_al_excluded=False,
            min_brightness=0,
            max_brightness=255,
            start="2026-04-01",
            end="2026-04-02",
            use_full_res_images=False,
            min_milliseconds_between_images=0,
        ),
        selection=SelectionSettings(
            strategy=strategy,
            available_strategies=[strategy],
            n_select=2,
            seed=42,
        ),
        models=[
            ModelOption(
                name="u-model", type="unet", path="/tmp/u.zip", image_size=(320, 240)
            ),
            ModelOption(
                name="a-model", type="unet", path="/tmp/a.zip", image_size=(320, 240)
            ),
        ],
        uncertainty_coreset=UncertaintySettings(
            feature_model="resnet50",
            uncertainty_model="u-model",
            alpha=0.5,
            provider="mc_dropout",
            mc_iterations=5,
            batch_size=11,
            aggregation="topk_mean",
            topk_fraction=0.1,
            candidate_multiplier=4,
        ),
        alges=AlgesSettings(
            model="a-model",
            method="semantic",
            batch_size=32,
        ),
        export=ExportSettings(
            sama_project_id=sama_project_id,
            sama_priority=sama_priority,
        ),
    )


def test_run_query_preview_builds_expected_response(monkeypatch):
    cfg = SimpleNamespace(
        models={
            "resnet50": SimpleNamespace(
                name="resnet50", type="resnet50", path=None, url=None
            ),
            "a-model": SimpleNamespace(
                name="a-model",
                type="unet",
                path="/tmp/a.zip",
                url="https://example.com/a.zip",
            ),
            "u-model": SimpleNamespace(
                name="u-model",
                type="unet",
                path="/tmp/u.zip",
                url="https://example.com/u.zip",
            ),
        },
        query=SimpleNamespace(
            cache_root="/tmp/cache",
            sama_priority=0,
            sensor="CAM_TROLLEY",
            exclude_seeded=True,
            exclude_al_excluded=False,
            min_brightness=12.0,
            max_brightness=180.0,
            brightness_filter_enabled=True,
            start="2026-04-01",
            end="2026-04-02",
            use_full_res_images=False,
            min_milliseconds_between_images=0.0,
        ),
        selection=SimpleNamespace(
            strategy="alges",
            n_select=200,
            seed=42,
        ),
        coreset=SimpleNamespace(
            feature_model="resnet50",
        ),
        uncertainty_coreset=SimpleNamespace(
            feature_model="resnet50",
            uncertainty_model="u-model",
            alpha=0.5,
            provider="mc_dropout",
            mc_iterations=5,
            batch_size=32,
            aggregation="topk_mean",
            topk_fraction=0.1,
            candidate_multiplier=4,
        ),
        alges=SimpleNamespace(
            model="a-model",
            method="semantic",
            batch_size=32,
        ),
        export=SimpleNamespace(
            sama_project_id=1,
        ),
    )
    artifacts = SimpleNamespace(
        cfg=cfg,
        all_ids=["a", "b", "c", "d"],
        labeled_ids=["a"],
        seeded_ids_count=0,
        pool_ids=["b", "c", "d"],
        candidate_ids=["b", "c"],
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.collect_query_preview_artifacts",
        lambda request, config_path=None, reporter=None: artifacts,
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.build_project_summary",
        lambda config_path, project_name: ProjectSummary(
            project_name=project_name,
            query_sensor="CAM_TROLLEY",
            sama_project_id=1,
            export_prefix="pref",
            selection_strategy="alges",
        ),
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.random.sample",
        lambda population, k: list(population)[:k],
    )

    response = run_query_preview(
        QueryPreviewRequest(project_name="demo", sample_size=2)
    )

    assert response.project.project_name == "demo"
    assert response.selection.n_select == 200
    assert response.all_ids_count == 4
    assert response.pool_ids_count == 3
    assert response.brightness_filtered_ids_count == 2
    assert response.preview_sample_ids == ["b", "c"]
    assert response.preview_items[0].thumbnail_url.endswith("b")


def test_apply_brightness_filter_skips_full_range(monkeypatch):
    cfg = SimpleNamespace(
        query=SimpleNamespace(
            min_brightness=0.0,
            max_brightness=255.0,
            cache_root="/tmp/cache",
        )
    )
    reporter = SimpleNamespace(skip_stages=lambda *stage_ids: stage_ids)
    monkeypatch.setattr(
        "active_learning.api.query_service.filter_by_brightness",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("filter_by_brightness should not be called")
        ),
    )

    sample_ids = ["a", "b", "c"]
    filtered_ids = apply_brightness_filter(
        cfg,
        provider=object(),
        sample_ids=sample_ids,
        reporter=reporter,
    )

    assert filtered_ids == sample_ids


def test_apply_brightness_filter_skips_when_disabled(monkeypatch):
    cfg = SimpleNamespace(
        query=SimpleNamespace(
            min_brightness=10.0,
            max_brightness=200.0,
            brightness_filter_enabled=False,
            cache_root="/tmp/cache",
        )
    )
    reporter = SimpleNamespace(skip_stages=lambda *stage_ids: stage_ids)
    monkeypatch.setattr(
        "active_learning.api.query_service.filter_by_brightness",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("filter_by_brightness should not be called")
        ),
    )

    sample_ids = ["a", "b", "c"]
    filtered_ids = apply_brightness_filter(
        cfg,
        provider=object(),
        sample_ids=sample_ids,
        reporter=reporter,
    )

    assert filtered_ids == sample_ids


def test_run_strategy_preview_builds_expected_response(monkeypatch):
    cfg = SimpleNamespace(
        models={
            "resnet50": SimpleNamespace(
                name="resnet50", type="resnet50", path=None, url=None
            ),
            "a-model": SimpleNamespace(
                name="a-model",
                type="unet",
                path="/tmp/a.zip",
                url="https://example.com/a.zip",
            ),
            "u-model": SimpleNamespace(
                name="u-model",
                type="unet",
                path="/tmp/u.zip",
                url="https://example.com/u.zip",
            ),
        },
        query=SimpleNamespace(
            cache_root="/tmp/cache",
            sama_priority=0,
            sensor="CAM_TROLLEY",
            exclude_seeded=True,
            exclude_al_excluded=False,
            min_brightness=12.0,
            max_brightness=180.0,
            brightness_filter_enabled=True,
            start="2026-04-01",
            end="2026-04-02",
            use_full_res_images=False,
            min_milliseconds_between_images=0.0,
        ),
        selection=SimpleNamespace(
            strategy="alges",
            n_select=200,
            seed=42,
        ),
        coreset=SimpleNamespace(
            feature_model="resnet50",
        ),
        uncertainty_coreset=SimpleNamespace(
            feature_model="resnet50",
            uncertainty_model="u-model",
            alpha=0.5,
            provider="mc_dropout",
            mc_iterations=5,
            batch_size=32,
            aggregation="topk_mean",
            topk_fraction=0.1,
            candidate_multiplier=4,
        ),
        alges=SimpleNamespace(
            model="a-model",
            method="semantic",
            batch_size=32,
        ),
        export=SimpleNamespace(
            sama_project_id=1,
        ),
    )
    artifacts = SimpleNamespace(
        cfg=cfg,
        all_ids=["a", "b", "c", "d"],
        labeled_ids=["a"],
        seeded_ids_count=0,
        pool_ids=["b", "c", "d"],
        candidate_ids=["b", "c"],
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.collect_query_preview_artifacts",
        lambda request, config_path=None, reporter=None: artifacts,
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.build_query_preview_config",
        lambda request, config_path=None, ensure_model_downloads=False: cfg,
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.build_project_summary",
        lambda config_path, project_name: ProjectSummary(
            project_name=project_name,
            query_sensor="CAM_TROLLEY",
            sama_project_id=1,
            export_prefix="pref",
            selection_strategy="alges",
        ),
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.get_crid",
        lambda: object(),
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.build_preview_provider",
        lambda cfg, crid: object(),
    )
    monkeypatch.setattr(
        "active_learning.api.query_service._prepare_strategy_models",
        lambda cfg: (None, None),
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.fetch_excluded_ids",
        lambda crid, sample_ids: set(),
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.random.sample",
        lambda population, k: list(population)[:k],
    )
    fake_selection_module = ModuleType("active_learning.core.selection")
    fake_selection_module.run_selection = lambda *args, **kwargs: SimpleNamespace(
        selected_ids=["b", "c"]
    )
    monkeypatch.setitem(
        sys.modules, "active_learning.core.selection", fake_selection_module
    )

    response = run_strategy_preview(
        QueryPreviewRequest(project_name="demo", sample_size=2)
    )

    assert response.result_kind == "strategy"
    assert response.selection.n_select == 200
    assert response.selected_ids_count == 2
    assert response.selected_ids == ["b", "c"]
    assert response.preview_sample_ids == ["b", "c"]
    assert response.overlay_available


def test_run_strategy_preview_includes_mask_urls_when_generated(monkeypatch):
    cfg = SimpleNamespace(
        models={
            "resnet50": SimpleNamespace(
                name="resnet50", type="resnet50", path=None, url=None
            ),
            "a-model": SimpleNamespace(
                name="a-model",
                type="unet",
                path="/tmp/a.zip",
                url="https://example.com/a.zip",
            ),
            "u-model": SimpleNamespace(
                name="u-model",
                type="unet",
                path="/tmp/u.zip",
                url="https://example.com/u.zip",
            ),
        },
        query=SimpleNamespace(
            sama_priority=0,
            sensor="CAM_TROLLEY",
            exclude_seeded=True,
            exclude_al_excluded=False,
            min_brightness=12.0,
            max_brightness=180.0,
            brightness_filter_enabled=True,
            start="2026-04-01",
            end="2026-04-02",
            use_full_res_images=False,
            min_milliseconds_between_images=0.0,
            cache_root="data/active_learning/feature_cache",
        ),
        selection=SimpleNamespace(
            strategy="uncertainty_coreset",
            n_select=200,
            seed=42,
        ),
        coreset=SimpleNamespace(feature_model="resnet50"),
        uncertainty_coreset=SimpleNamespace(
            feature_model="resnet50",
            uncertainty_model="u-model",
            alpha=0.5,
            provider="mc_dropout",
            mc_iterations=5,
            batch_size=32,
            aggregation="topk_mean",
            topk_fraction=0.1,
            candidate_multiplier=4,
        ),
        alges=SimpleNamespace(
            model="a-model",
            method="semantic",
            batch_size=32,
        ),
        export=SimpleNamespace(
            sama_project_id=1,
        ),
    )
    artifacts = SimpleNamespace(
        cfg=cfg,
        all_ids=["a", "b", "c", "d"],
        labeled_ids=["a"],
        seeded_ids_count=0,
        pool_ids=["b", "c", "d"],
        candidate_ids=["b", "c"],
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.collect_query_preview_artifacts",
        lambda request, config_path=None, reporter=None: artifacts,
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.build_query_preview_config",
        lambda request, config_path=None, ensure_model_downloads=False: cfg,
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.build_project_summary",
        lambda config_path, project_name: ProjectSummary(
            project_name=project_name,
            query_sensor="CAM_TROLLEY",
            sama_project_id=1,
            export_prefix="pref",
            selection_strategy="uncertainty_coreset",
        ),
    )
    monkeypatch.setattr("active_learning.api.query_service.get_crid", lambda: object())
    monkeypatch.setattr(
        "active_learning.api.query_service.build_preview_provider",
        lambda cfg, crid: object(),
    )
    monkeypatch.setattr(
        "active_learning.api.query_service._prepare_strategy_models",
        lambda cfg: (None, None),
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.fetch_excluded_ids",
        lambda crid, sample_ids: set(),
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.generate_preview_masks",
        lambda cfg, provider, sample_ids: {
            sid: f"/api/al/masks/{sid}" for sid in sample_ids
        },
    )
    fake_selection_module = ModuleType("active_learning.core.selection")
    fake_selection_module.run_selection = lambda *args, **kwargs: SimpleNamespace(
        selected_ids=["b", "c"]
    )
    monkeypatch.setitem(
        sys.modules, "active_learning.core.selection", fake_selection_module
    )

    response = run_strategy_preview(
        QueryPreviewRequest(project_name="demo", sample_size=2)
    )

    assert response.preview_items[0].mask_url == "/api/al/masks/b"


def test_build_overlay_mosaic_download_uses_posted_export_context(monkeypatch):
    export_context = make_export_context(
        strategy="uncertainty_coreset", sama_project_id=None, sama_priority=0
    )
    monkeypatch.setattr("active_learning.api.query_service.get_crid", lambda: object())
    monkeypatch.setattr(
        "active_learning.api.query_service.build_preview_provider",
        lambda cfg, crid: "provider",
    )
    captured = {}

    def fake_render_overlay_mosaic(result, provider, *, cfg, progress):
        captured["selected_ids"] = list(result.selected_ids)
        captured["provider"] = provider
        captured["model"] = cfg.alges.model
        captured["batch_size"] = cfg.alges.batch_size
        return "/tmp/overlay.jpg"

    monkeypatch.setattr(
        "active_learning.api.query_service.render_overlay_mosaic",
        fake_render_overlay_mosaic,
    )

    response = build_overlay_mosaic_download(export_context, selected_ids=["y"])

    assert response.path == "/tmp/overlay.jpg"
    assert captured == {
        "selected_ids": ["y"],
        "provider": "provider",
        "model": "u-model",
        "batch_size": 11,
    }


def test_seed_strategy_selection_uses_posted_export_context(monkeypatch):
    export_context = make_export_context(
        strategy="alges", sama_project_id=12, sama_priority=7
    )
    monkeypatch.setattr(
        "active_learning.api.query_service.build_export_description",
        lambda result, cfg, strategy: (
            "export-demo",
            f"desc-{strategy}-{len(result.selected_ids)}",
        ),
    )
    monkeypatch.setattr("active_learning.api.query_service.get_crid", lambda: "crid")

    class FakeSink:
        def __init__(self, crid, *, sama_project_id, priority, overwrite):
            assert crid == "crid"
            assert sama_project_id == 12
            assert priority == 7
            assert not overwrite

        def submit(self, result, *, export_id, description):
            assert list(result.selected_ids) == ["a", "c"]
            assert export_id == "export-demo"
            assert description == "desc-alges-2"
            return SimpleNamespace(
                export_id=export_id,
                description=description,
                sama_batch_id="batch-99",
                image_count=2,
            )

    monkeypatch.setattr("active_learning.api.query_service.CridSamaSink", FakeSink)

    response = seed_strategy_selection(export_context, selected_ids=["a", "c"])

    assert response.export_id == "export-demo"
    assert response.sama_batch_id == "batch-99"
    assert response.image_count == 2
