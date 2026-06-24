"""Small per-sample cache helpers for scorer modules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from active_learning.core.types import SampleId


def _safe_fragment(sample_id: SampleId) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in str(sample_id))
    sanitized = sanitized.strip("_")
    return sanitized[:48] or "sample"


def _sample_digest(sample_id: SampleId) -> str:
    return hashlib.sha1(sample_id.encode("utf-8")).hexdigest()


def namespace_root(cache_root: str | Path, namespace: str) -> Path:
    root = Path(cache_root) / namespace
    root.mkdir(parents=True, exist_ok=True)
    return root


def sample_json_path(
    cache_root: str | Path,
    namespace: str,
    sample_id: SampleId,
) -> Path:
    digest = _sample_digest(sample_id)
    root = namespace_root(cache_root, namespace) / digest[:2]
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{digest}_{_safe_fragment(sample_id)}.json"


def sample_npy_path(
    cache_root: str | Path,
    namespace: str,
    sample_id: SampleId,
) -> Path:
    digest = _sample_digest(sample_id)
    root = namespace_root(cache_root, namespace) / digest[:2]
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{digest}_{_safe_fragment(sample_id)}.npy"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


def load_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
