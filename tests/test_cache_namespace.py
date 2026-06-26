"""Tests for cache namespacing helpers."""

import pytest

from active_learning.providers.unet import _resolve_model_path, unet_cache_namespace


class _DummyUnet:
    def __init__(self, config):
        self._config = config

    def get_config(self):
        return self._config


def test_unet_cache_namespace_changes_with_config():
    a = _DummyUnet({"input_shape": (320, 240, 3), "num_classes": 4})
    b = _DummyUnet({"input_shape": (320, 240, 3), "num_classes": 8})
    ns_a = unet_cache_namespace(a, "poeppelmann_only")
    ns_b = unet_cache_namespace(b, "poeppelmann_only")
    assert ns_a != ns_b
    assert ns_a.startswith("poeppelmann_only_")
    assert ns_b.startswith("poeppelmann_only_")


def test_resolve_model_path_accepts_file_path(tmp_path):
    model_zip = tmp_path / "model.zip"
    model_zip.write_bytes(b"not really a model")

    assert _resolve_model_path(str(model_zip)) == str(model_zip)


def test_resolve_model_path_uses_sibling_zip_for_legacy_directory(tmp_path):
    model_dir = tmp_path / "unet_260220_0000000_custom_fold1"
    model_dir.mkdir()
    model_zip = tmp_path / "unet_260220_0000000_custom_fold1.zip"
    model_zip.write_bytes(b"not really a model")

    assert _resolve_model_path(str(model_dir)) == str(model_zip)


def test_resolve_model_path_uses_single_zip_inside_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_zip = model_dir / "fold1.zip"
    model_zip.write_bytes(b"not really a model")

    assert _resolve_model_path(str(model_dir)) == str(model_zip)


def test_resolve_model_path_rejects_ambiguous_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "fold1.zip").write_bytes(b"not really a model")
    (model_dir / "fold2.zip").write_bytes(b"not really a model")

    with pytest.raises(IsADirectoryError, match="multiple .zip files"):
        _resolve_model_path(str(model_dir))
