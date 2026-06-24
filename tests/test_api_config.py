from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from active_learning.api import config_service
from active_learning.api.schemas import SaveProjectConfigRequest


def test_list_projects_returns_summaries(monkeypatch):
    monkeypatch.delenv("ORIGIN", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "al_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "projects:",
                    "  demo:",
                    "    query:",
                    "      sensor: CAM_TROLLEY",
                    "    export:",
                    "      sama_project_id: 123",
                    "      prefix: demo-prefix",
                    "selection:",
                    "  strategy: alges",
                    "models:",
                    "  model_a:",
                    "    type: unet",
                    "    path: /tmp/model.zip",
                    "alges:",
                    "  model: model_a",
                ]
            ),
            encoding="utf-8",
        )
        response = config_service.list_projects(config_path)
        assert response.projects[0].project_name == "demo"
        assert response.projects[0].query_sensor == "CAM_TROLLEY"
        assert response.projects[0].sama_project_id == 123


def test_list_projects_filters_by_active_origin(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "al_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "projects:",
                    "  miniportal:",
                    "    query:",
                    "      sensor: CAM_TROLLEY",
                    "    export:",
                    "      sama_project_id: 1",
                    "  poeppelmann:",
                    "    crid_origin: miniportal",
                    "    query:",
                    "      sensor: CAM_TROLLEY_HD",
                    "    export:",
                    "      sama_project_id: 2",
                    "  skasoegel:",
                    "    query:",
                    "      sensor: CAM_TROLLEY",
                    "    export:",
                    "      sama_project_id: 3",
                    "selection:",
                    "  strategy: alges",
                    "models:",
                    "  model_a:",
                    "    type: unet",
                    "    path: /tmp/model.zip",
                    "alges:",
                    "  model: model_a",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("ORIGIN", "miniportal")
        response = config_service.list_projects(config_path)
        assert [project.project_name for project in response.projects] == [
            "miniportal",
            "poeppelmann",
        ]


def test_get_project_config_returns_query_defaults(monkeypatch):
    monkeypatch.delenv("ORIGIN", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "al_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "projects:",
                    "  demo:",
                    "    query:",
                    "      sensor: CAM_TROLLEY",
                    "    export:",
                    "      sama_project_id: 123",
                    "models:",
                    "  model_a:",
                    "    type: unet",
                    "    path: /tmp/model.zip",
                    "selection:",
                    "  strategy: alges",
                    "query:",
                    "  exclude_seeded: true",
                    "  min_brightness: 10",
                    "  max_brightness: 200",
                    "  start: '2026-01-01'",
                    "  end: '2026-01-31'",
                    "alges:",
                    "  model: model_a",
                ]
            ),
            encoding="utf-8",
        )
        response = config_service.get_project_config("demo", config_path)
        assert response.project.project_name == "demo"
        assert response.query.cache_root
        assert response.query.exclude_seeded is True
        assert response.query.min_brightness == 10


def test_get_project_config_rejects_project_for_other_origin(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "al_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "projects:",
                    "  miniportal:",
                    "    query:",
                    "      sensor: CAM_TROLLEY",
                    "  skasoegel:",
                    "    query:",
                    "      sensor: CAM_TROLLEY",
                    "models:",
                    "  model_a:",
                    "    type: unet",
                    "    path: /tmp/model.zip",
                    "selection:",
                    "  strategy: alges",
                    "alges:",
                    "  model: model_a",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("ORIGIN", "miniportal")
        with pytest.raises(ValueError, match="not available"):
            config_service.get_project_config("skasoegel", config_path)


def test_save_project_config_updates_top_level_model_url_from_models_patch():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "al_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "projects:",
                    "  demo:",
                    "    selection:",
                    "      strategy: uncertainty_topk_coreset",
                    "    uncertainty_coreset:",
                    "      uncertainty_model: shared_model",
                    "    alges:",
                    "      model: shared_model",
                    "models:",
                    "  shared_model:",
                    "    type: unet",
                    "    url: old-url",
                    "selection:",
                    "  strategy: alges",
                    "uncertainty_coreset:",
                    "  feature_model: resnet50",
                    "  uncertainty_model: shared_model",
                    "  alpha: 0.5",
                    "  provider: bald",
                    "  mc_iterations: 5",
                    "  batch_size: 32",
                    "  aggregation: topk_mean",
                    "  topk_fraction: 0.1",
                    "  candidate_multiplier: 4",
                    "alges:",
                    "  model: shared_model",
                    "  method: semantic",
                    "  batch_size: 32",
                ]
            ),
            encoding="utf-8",
        )

        config_service.save_project_config(
            "demo",
            SaveProjectConfigRequest(
                uncertainty_coreset={"uncertainty_model": "shared_model", "alpha": 0.7},
                alges={"model": "shared_model", "method": "image"},
                models={"shared_model": {"url": "new-url"}},
            ),
            config_path,
        )

        yaml = YAML(typ="safe")
        raw = yaml.load(config_path.read_text(encoding="utf-8"))
        assert raw["projects"]["demo"]["uncertainty_coreset"]["alpha"] == 0.7
        assert raw["projects"]["demo"]["alges"]["method"] == "image"
        assert raw["models"]["shared_model"]["url"] == "new-url"
