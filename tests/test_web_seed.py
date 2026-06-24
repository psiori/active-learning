from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from active_learning.integrations.crid.export import (
    CridExportResult,
    build_export_description,
    export_selection,
)
from active_learning.integrations.crid.provider_source import CridImageProviderSource
from active_learning.core.image_provider import ImageProvider
from active_learning.sinks.mosaic import mosaic_output_path
from active_learning.sinks.mosaic import render_mosaic
from active_learning.sinks.sama import CridSamaSink
from active_learning.core.types import SelectionResult


class WebSeedSession:
    def __init__(
        self,
        crid,
        cfg,
        payload: dict,
        sample_ids: list[str],
        input_path: str,
    ):
        self.crid = crid
        self.cfg = cfg
        self.payload = payload
        self.input_path = input_path
        self.sample_ids = list(sample_ids)
        self.selected = [True] * len(sample_ids)
        self.tiles = []

    def toggle(self, index: int) -> None:
        self.selected[index] = not self.selected[index]

    def _build_tiles(self) -> None:
        paths = preview_paths_for_sample_ids(self.crid, self.cfg, self.sample_ids)
        self.tiles = [
            {"sample_id": sample_id, "source_path": path, "ready": True}
            for sample_id, path in zip(self.sample_ids, paths, strict=True)
        ]

    def snapshot(self) -> dict:
        if not self.tiles:
            self._build_tiles()
        return {
            "selected_count": sum(self.selected),
            "total_count": len(self.sample_ids),
            "tiles": [
                {
                    **tile,
                    "selected": self.selected[index],
                }
                for index, tile in enumerate(self.tiles)
            ],
        }

    def seed_selected(self) -> dict:
        result = SelectionResult(
            selected_ids=[
                sample_id
                for sample_id, is_selected in zip(
                    self.sample_ids,
                    self.selected,
                    strict=True,
                )
                if is_selected
            ],
        )
        mosaic_path = render_mosaic(
            result,
            self._provider(),
            mosaic_output_path(self.cfg),
            use_highres=self.cfg.query.use_full_res_images,
            resize_height=150,
            max_images=200,
            rows=10,
            cols=20,
        )
        export_id, description = build_export_description(
            result,
            self.cfg,
            self.cfg.selection.strategy,
        )
        if self.cfg.export.sama_project_id:
            final_export = CridSamaSink(
                self.crid,
                sama_project_id=self.cfg.export.sama_project_id,
                priority=self.cfg.query.sama_priority,
                overwrite=False,
            ).submit(result, export_id=export_id, description=description)
        else:
            export_result = export_selection(
                self.crid,
                result,
                export_id=export_id,
                description=description,
                overwrite=False,
            )
            final_export = CridExportResult(
                export_id=export_result.export_id,
                description=export_result.description,
            )
        return {
            "mosaic_path": mosaic_path,
            "export_id": final_export.export_id,
            "description": final_export.description,
            "sama_batch_id": getattr(final_export, "sama_batch_id", None),
            "image_count": len(result.selected_ids),
        }

    def _provider(self) -> ImageProvider:
        source = CridImageProviderSource(
            self.crid,
            Path(self.cfg.query.cache_root) / "web_seed_source",
        )
        return ImageProvider(
            source,
            cache_root=Path(self.cfg.query.cache_root) / "web_seed_provider",
        )


def preview_paths_for_sample_ids(crid, cfg, sample_ids: list[str]) -> list[str]:
    source = CridImageProviderSource(
        crid,
        Path(cfg.query.cache_root) / "web_seed_source",
    )
    provider = ImageProvider(
        source,
        cache_root=Path(cfg.query.cache_root) / "web_seed_provider",
    )
    if cfg.query.use_full_res_images:
        return provider.get_highres_batch(sample_ids, progress=True)
    return provider.get_lowres_batch(sample_ids, progress=True)


class FakeCRID:
    def __init__(self):
        self.exports = []
        self.azure_client = SimpleNamespace(
            config=SimpleNamespace(account_url="https://example.blob.core.windows.net"),
            get_sas_token=lambda container, expiry_in_days=7: "sig=fake",
        )
        self.config = SimpleNamespace(
            azure=SimpleNamespace(data_container="images"),
        )
        self.dataset_service = SimpleNamespace(
            _create_blob_url_lambda=lambda account_url, container, sas_token: (
                lambda blob: f"{account_url}/{container}/{blob}?{sas_token}"
            ),
            _create_blob_url_no_sas_lambda=lambda account_url, container: (
                lambda blob: f"{account_url}/{container}/{blob}"
            ),
        )

    def export_dataset(self, dataset, export_id, description, overwrite=False):
        self.exports.append((list(dataset.blob_ids), export_id, description))


class FakeProvider:
    def __init__(self, tmpdir: str):
        self.tmpdir = Path(tmpdir)

    def get_lowres_batch(self, sample_ids, *, progress=False, allow_missing=False):
        paths = []
        for sample_id in sample_ids:
            path = self.tmpdir / f"{sample_id}.png"
            Image.new("RGB", (32, 32), color=(80, 10, 10)).save(path)
            paths.append(str(path))
        return paths

    def get_highres_batch(self, sample_ids, *, progress=False, allow_missing=False):
        return self.get_lowres_batch(
            sample_ids,
            progress=progress,
            allow_missing=allow_missing,
        )


def _cfg(tmpdir: str):
    return SimpleNamespace(
        query=SimpleNamespace(
            cache_root=tmpdir,
            start="2026-04-01",
            end="2026-04-02",
            use_full_res_images=False,
            sama_priority=0,
        ),
        export=SimpleNamespace(
            project="proj",
            prefix="pref",
            sama_project_id=None,
            mosaic_path=str(Path(tmpdir) / "out.png"),
        ),
        selection=SimpleNamespace(n_select=2, strategy="coreset", seed=7),
        uncertainty_coreset=SimpleNamespace(
            alpha=0.5,
            provider="bald",
            aggregation="topk_mean",
        ),
    )


def test_session_selection_and_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        session = WebSeedSession(
            FakeCRID(),
            cfg,
            {"sample_ids": ["a", "b"]},
            ["a", "b"],
            str(Path(tmp) / "in.yaml"),
        )
        Image.new("RGB", (32, 32), color=(10, 10, 10)).save(Path(tmp) / "a.png")
        Image.new("RGB", (32, 32), color=(10, 10, 10)).save(Path(tmp) / "b.png")
        with patch(
            "test_web_seed.preview_paths_for_sample_ids",
            return_value=[str(Path(tmp) / "a.png"), str(Path(tmp) / "b.png")],
        ):
            session._build_tiles()
        session.toggle(1)
        snap = session.snapshot()
        assert snap["selected_count"] == 1
        assert len(snap["tiles"]) == 2
        assert not snap["tiles"][1]["selected"]


def test_seed_selected_calls_export_and_mosaic():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        crid = FakeCRID()
        session = WebSeedSession(
            crid,
            cfg,
            {"sample_ids": ["a", "b"]},
            ["a", "b"],
            str(Path(tmp) / "in.yaml"),
        )
        session.selected = [True, False]
        with patch.object(WebSeedSession, "_provider", return_value=FakeProvider(tmp)):
            result = session.seed_selected()
        assert result["mosaic_path"].endswith(".jpg")
        assert crid.exports[0][0] == ["a"]


def test_seed_selected_uses_crid_sama_sink_when_sama_project_is_set():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        cfg.export.sama_project_id = 123
        crid = FakeCRID()
        session = WebSeedSession(
            crid,
            cfg,
            {"sample_ids": ["a", "b"]},
            ["a", "b"],
            str(Path(tmp) / "in.yaml"),
        )
        session.selected = [True, False]

        class FakeSink:
            def __init__(
                self,
                crid_arg,
                *,
                sama_project_id,
                priority=0,
                overwrite=False,
            ):
                assert crid_arg is crid
                assert sama_project_id == 123
                assert priority == 0
                assert overwrite is False

            def submit(self, result, *, export_id, description):
                assert result.selected_ids == ["a"]
                return SimpleNamespace(
                    export_id=export_id,
                    description=description,
                    sama_batch_id="batch-9",
                    image_count=1,
                )

        with (
            patch.object(WebSeedSession, "_provider", return_value=FakeProvider(tmp)),
            patch.object(__import__(__name__), "CridSamaSink", FakeSink),
        ):
            result = session.seed_selected()

        assert result["sama_batch_id"] == "batch-9"
