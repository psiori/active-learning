from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


QUERY_CACHE_VERSION = 1
QUERY_CACHE_TTL_SECONDS = 24 * 60 * 60
QUERY_CACHE_MAX_SESSIONS = 10
QUERY_CACHE_MAX_BYTES = 500 * 1024 * 1024
CACHED_QUERY_STAGE_IDS = (
    "query_pool",
    "query_labeled",
    "fetch_seeded",
    "time_gap_filter",
    "brightness_filter",
)


@dataclass(slots=True)
class QueryCacheEntry:
    token: str
    fingerprint: str
    created_at: float
    last_accessed_at: float
    size_bytes: int
    path: str
    version: int = QUERY_CACHE_VERSION


class QueryCacheStore:
    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root)
        self.base_dir = self.cache_root / "webapp_query_cache"
        self.sessions_dir = self.base_dir / "sessions"
        self.index_path = self.base_dir / "index.json"
        self._lock = threading.Lock()

    def save(self, fingerprint: str, payload: dict[str, Any]) -> str:
        now = time.time()
        token = uuid.uuid4().hex
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self.sessions_dir / f"{token}.json.gz"

        body = {
            "version": QUERY_CACHE_VERSION,
            "token": token,
            "fingerprint": fingerprint,
            "created_at": now,
            "last_accessed_at": now,
            "payload": payload,
        }

        with self._lock:
            self._write_session(path, body)
            size_bytes = path.stat().st_size
            entries = self._load_index()
            entries[token] = QueryCacheEntry(
                token=token,
                fingerprint=fingerprint,
                created_at=now,
                last_accessed_at=now,
                size_bytes=size_bytes,
                path=str(path),
            )
            self._prune_locked(entries, now=now)
            self._write_index(entries)
        return token

    def load(self, token: str, fingerprint: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            entries = self._load_index()
            entry = entries.get(token)
            if entry is None:
                return None
            if self._is_expired(entry, now=now) or entry.version != QUERY_CACHE_VERSION:
                self._delete_entry_file(entry)
                entries.pop(token, None)
                self._prune_locked(entries, now=now)
                self._write_index(entries)
                return None
            if entry.fingerprint != fingerprint:
                return None

            path = Path(entry.path)
            payload = self._read_session(path)
            if payload is None:
                entries.pop(token, None)
                self._prune_locked(entries, now=now)
                self._write_index(entries)
                return None
            if (
                payload.get("version") != QUERY_CACHE_VERSION
                or payload.get("fingerprint") != fingerprint
            ):
                return None

            entry.last_accessed_at = now
            if path.exists():
                entry.size_bytes = path.stat().st_size
            self._write_session(
                path,
                {
                    **payload,
                    "last_accessed_at": now,
                },
            )
            entries[token] = entry
            self._prune_locked(entries, now=now)
            self._write_index(entries)
            return payload.get("payload")

    def _load_index(self) -> dict[str, QueryCacheEntry]:
        if not self.index_path.exists():
            return self._rebuild_index()
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._rebuild_index()
        version = data.get("version")
        if version != QUERY_CACHE_VERSION:
            return self._rebuild_index()
        entries: dict[str, QueryCacheEntry] = {}
        for raw in data.get("entries", []):
            try:
                entry = QueryCacheEntry(**raw)
            except TypeError:
                continue
            entries[entry.token] = entry
        return entries

    def _rebuild_index(self) -> dict[str, QueryCacheEntry]:
        entries: dict[str, QueryCacheEntry] = {}
        if not self.sessions_dir.exists():
            return entries
        for path in self.sessions_dir.glob("*.json.gz"):
            payload = self._read_session(path)
            if payload is None:
                self._safe_unlink(path)
                continue
            if payload.get("version") != QUERY_CACHE_VERSION:
                self._safe_unlink(path)
                continue
            token = payload.get("token")
            fingerprint = payload.get("fingerprint")
            created_at = payload.get("created_at")
            last_accessed_at = payload.get("last_accessed_at", created_at)
            if not token or not fingerprint or created_at is None:
                self._safe_unlink(path)
                continue
            entries[token] = QueryCacheEntry(
                token=token,
                fingerprint=fingerprint,
                created_at=float(created_at),
                last_accessed_at=float(last_accessed_at),
                size_bytes=path.stat().st_size,
                path=str(path),
            )
        return entries

    def _prune_locked(self, entries: dict[str, QueryCacheEntry], *, now: float) -> None:
        for token, entry in list(entries.items()):
            if self._is_expired(entry, now=now):
                self._delete_entry_file(entry)
                entries.pop(token, None)

        while len(entries) > QUERY_CACHE_MAX_SESSIONS:
            token, entry = self._next_eviction_candidate(entries)
            self._delete_entry_file(entry)
            entries.pop(token, None)

        while self._total_size(entries) > QUERY_CACHE_MAX_BYTES and entries:
            token, entry = self._next_eviction_candidate(entries)
            self._delete_entry_file(entry)
            entries.pop(token, None)

    def _write_index(self, entries: dict[str, QueryCacheEntry]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": QUERY_CACHE_VERSION,
            "entries": [asdict(entry) for entry in entries.values()],
        }
        tmp_path = self.index_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        os.replace(tmp_path, self.index_path)

    def _is_expired(self, entry: QueryCacheEntry, *, now: float) -> bool:
        return now - entry.last_accessed_at > QUERY_CACHE_TTL_SECONDS

    @staticmethod
    def _next_eviction_candidate(
        entries: dict[str, QueryCacheEntry],
    ) -> tuple[str, QueryCacheEntry]:
        return min(
            entries.items(),
            key=lambda item: (
                item[1].last_accessed_at,
                item[1].created_at,
                item[0],
            ),
        )

    @staticmethod
    def _total_size(entries: dict[str, QueryCacheEntry]) -> int:
        return sum(entry.size_bytes for entry in entries.values())

    def invalidate_all(self) -> None:
        with self._lock:
            entries = self._load_index()
            for entry in list(entries.values()):
                self._delete_entry_file(entry)
            self._write_index({})

    def _delete_entry_file(self, entry: QueryCacheEntry) -> None:
        self._safe_unlink(Path(entry.path))

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    @staticmethod
    def _write_session(path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(tmp_path, path)

    @staticmethod
    def _read_session(path: Path) -> dict[str, Any] | None:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None


def build_query_fingerprint(cfg, project_name: str) -> str:
    payload = {
        "version": QUERY_CACHE_VERSION,
        "project_name": project_name,
        "sensor": cfg.query.sensor,
        "sama_project_id": cfg.export.sama_project_id,
        "exclude_seeded": cfg.query.exclude_seeded,
        "exclude_al_excluded": cfg.query.exclude_al_excluded,
        "min_brightness": cfg.query.min_brightness,
        "max_brightness": cfg.query.max_brightness,
        "brightness_filter_enabled": cfg.query.brightness_filter_enabled,
        "start": cfg.query.start,
        "end": cfg.query.end,
        "use_full_res_images": cfg.query.use_full_res_images,
        "min_milliseconds_between_images": cfg.query.min_milliseconds_between_images,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def serialize_query_artifacts(artifacts) -> dict[str, Any]:
    return {
        "all_ids": list(artifacts.all_ids),
        "labeled_ids": list(artifacts.labeled_ids),
        "seeded_ids_count": int(artifacts.seeded_ids_count),
        "pool_ids": list(artifacts.pool_ids),
        "candidate_ids": list(artifacts.candidate_ids),
    }
