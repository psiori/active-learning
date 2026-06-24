from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.responses import FileResponse

from active_learning.api.query_service import build_preview_provider, get_crid


def get_thumbnail_path(
    sample_id: str,
    *,
    cache_root: str,
    use_full_res_images: bool = False,
) -> Path:
    crid = get_crid()
    cfg = SimpleNamespace(
        query=SimpleNamespace(
            cache_root=cache_root,
            use_full_res_images=use_full_res_images,
        )
    )
    provider = build_preview_provider(cfg, crid)
    path = (
        provider.get_highres(sample_id)
        if use_full_res_images
        else provider.get_lowres(sample_id)
    )
    return Path(path)


def build_thumbnail_response(
    sample_id: str,
    *,
    cache_root: str,
    use_full_res_images: bool = False,
) -> FileResponse:
    path = get_thumbnail_path(
        sample_id,
        cache_root=cache_root,
        use_full_res_images=use_full_res_images,
    )
    return FileResponse(path)
