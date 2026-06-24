from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image
from ruamel.yaml import YAML

from active_learning.core.types import SelectionResult
from active_learning.local import main


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def test_local_cli_writes_mosaic_and_yaml_for_directory(tmp_path):
    images_dir = tmp_path / "images"
    _write_image(images_dir / "a.jpg", (10, 20, 30))
    _write_image(images_dir / "nested" / "b.PNG", (30, 20, 10))
    config_path = tmp_path / "local.yaml"
    mosaic_path = tmp_path / "out" / "selection.jpg"
    cache_root = tmp_path / "cache"
    config_path.write_text(
        f"""
query:
  cache_root: {cache_root}
  brightness_filter_enabled: false
selection:
  strategy: coreset
  n_select: 2
  seed: 7
export:
  mosaic_path: {mosaic_path}
""",
        encoding="utf-8",
    )

    with (
        patch("active_learning.local.describe_tensorflow_device", return_value="tf"),
        patch(
            "active_learning.local.select_local_samples",
            return_value=SelectionResult(
                selected_ids=["a.jpg", "nested/b.PNG"],
                details={"selector": "test"},
            ),
        ) as select_mock,
    ):
        main(["--images-dir", str(images_dir), "--config", str(config_path)])

    assert select_mock.call_args.args[2] == ["a.jpg", "nested/b.PNG"]
    assert mosaic_path.exists()
    payload = YAML(typ="safe").load(mosaic_path.with_suffix(".yaml"))
    assert payload["kind"] == "local_selection"
    assert payload["source"] == {
        "type": "local_directory",
        "images_dir": str(images_dir.resolve()),
    }
    assert payload["selection"] == {
        "strategy": "coreset",
        "n_select": 2,
        "seed": 7,
    }
    assert payload["sample_ids"] == ["a.jpg", "nested/b.PNG"]
