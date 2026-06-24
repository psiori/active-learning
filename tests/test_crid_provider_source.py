"""Tests for CRID image provider staging downloads."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from azure.common import AzureHttpError
from PIL import Image

from active_learning.integrations.crid.provider_source import CridImageProviderSource
from interface.service.dataset import DatasetService


def _miniportal_blob() -> str:
    return (
        "miniportal/images/trolley0/cam_trolley_right/2022/08/18/"
        "2022-08-18T14_41_56_103Z.png"
    )


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_staging_reuses_readable_file_without_azure_call(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIGIN", "miniportal")
    blob = _miniportal_blob()
    dl = tmp_path / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    local_path = dl / DatasetService._get_filename(blob)
    Image.new("RGB", (2, 2)).save(local_path, format="PNG")

    calls = []

    class FakeAzure:
        def get_blob_to_stream(self, *args, **kwargs):
            calls.append(1)

    crid = SimpleNamespace(
        azure_client=FakeAzure(),
        config=SimpleNamespace(azure=SimpleNamespace(data_container="data")),
    )
    src = CridImageProviderSource(crid, dl)
    out = src._download_blob(blob)
    assert out == str(local_path)
    assert calls == []


def test_staging_redownloads_when_existing_file_is_not_a_valid_image(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ORIGIN", "miniportal")
    blob = _miniportal_blob()
    dl = tmp_path / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    local_path = dl / DatasetService._get_filename(blob)
    local_path.write_bytes(b"truncated-or-stale")

    png = _png_bytes()
    calls = []

    class FakeAzure:
        def get_blob_to_stream(self, _container, _blob, fp):
            calls.append(1)
            fp.write(png)

    crid = SimpleNamespace(
        azure_client=FakeAzure(),
        config=SimpleNamespace(azure=SimpleNamespace(data_container="data")),
    )
    src = CridImageProviderSource(crid, dl)
    out = src._download_blob(blob)
    assert calls == [1]
    assert out == str(local_path)
    with Image.open(out) as im:
        im.load()


def test_azure_blob_not_found_maps_to_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIGIN", "miniportal")
    blob = _miniportal_blob()
    dl = tmp_path / "dl"
    dl.mkdir(parents=True, exist_ok=True)

    class FakeAzure:
        def get_blob_to_stream(self, *args, **kwargs):
            raise AzureHttpError(
                "The specified blob does not exist. ErrorCode: BlobNotFound", 404
            )

    crid = SimpleNamespace(
        azure_client=FakeAzure(),
        config=SimpleNamespace(azure=SimpleNamespace(data_container="data")),
    )
    src = CridImageProviderSource(crid, dl)
    with pytest.raises(FileNotFoundError, match="Blob not found in Azure"):
        src._download_blob(blob, sample_id="s1")


def test_azure_non_404_error_propagates(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIGIN", "miniportal")
    blob = _miniportal_blob()
    dl = tmp_path / "dl"
    dl.mkdir(parents=True, exist_ok=True)

    class FakeAzure:
        def get_blob_to_stream(self, *args, **kwargs):
            raise AzureHttpError("Server error", 500)

    crid = SimpleNamespace(
        azure_client=FakeAzure(),
        config=SimpleNamespace(azure=SimpleNamespace(data_container="data")),
    )
    src = CridImageProviderSource(crid, dl)
    with pytest.raises(AzureHttpError) as excinfo:
        src._download_blob(blob, sample_id="s1")
    assert excinfo.value.status_code == 500
