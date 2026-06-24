#!/usr/bin/env python3
"""List Elasticsearch documents that share the same logical blob path as the main seed entrypoint's pool query.

Run from ``apps/crid`` with the same ``-c`` / ``-p`` / date flags as the main seed entrypoint. Uses the
same CRID filter and ``max_size`` as :meth:`active_learning.integrations.crid.source.CridSource.query_ids`
(no brightness filter, no seeded exclusion — raw ES rows only).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

from active_learning.core.config import (
    SEED_CLI_MAP,
    _UNSET,
    build_seed_config,
    load_yaml,
    merge_cli,
    parse_query_datetime,
    resolve_project,
)
from interface.crid import CRID
from interface.model.models import CRIDFilter, SensorName
from interface.service.dataset import DatasetService

for key, value in list(os.environ.items()):
    if key.startswith("ENV_"):
        os.environ[key[4:]] = value


def _augment_unset_args(ns: argparse.Namespace) -> None:
    for name in SEED_CLI_MAP:
        if not hasattr(ns, name):
            setattr(ns, name, _UNSET)


def _logical_blob_from_hit(hit: dict) -> str | None:
    src = hit.get("_source") or {}
    iu = src.get("image_url")
    if iu:
        return DatasetService._image_url_to_blob(iu)
    blob = src.get("blob")
    if isinstance(blob, str) and blob:
        return DatasetService._logical_image_blob_path(blob)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show ES documents whose logical blob path appears more than once "
        "(same query shape as CridSource.query_ids).",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to YAML (e.g. active_learning/al_config.yaml)",
    )
    parser.add_argument("-p", "--project", default=_UNSET, help="Project preset name")
    parser.add_argument("--start", default=_UNSET, help="Query window start (ISO)")
    parser.add_argument("--end", default=_UNSET, help="Query window end (ISO)")
    parser.add_argument(
        "-s",
        "--sensor",
        default=_UNSET,
        help="Override sensor enum name (default from merged config)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=_UNSET,
        help="Max ES hits (default: 1000000, same as CridSource.query_ids)",
    )
    args = parser.parse_args()
    _augment_unset_args(args)

    yaml_dict = load_yaml(args.config)
    yaml_dict = resolve_project(
        yaml_dict,
        args.project if args.project is not _UNSET else None,
    )
    merged = merge_cli(yaml_dict, args, SEED_CLI_MAP)
    cfg = build_seed_config(merged, ensure_model_downloads=False)

    sensor_name = getattr(SensorName, cfg.query.sensor)
    start_dt = parse_query_datetime(cfg.query.start)
    end_dt = parse_query_datetime(cfg.query.end)
    max_size = 1_000_000
    if args.max_size is not _UNSET:
        max_size = int(args.max_size)

    crid_filter = CRIDFilter()
    crid_filter.sensor_name = sensor_name
    crid_filter.start_datetime = start_dt
    crid_filter.end_datetime = end_dt

    crid = CRID(globals())
    es_result = crid.elasticsearch_client.query(crid_filter, max_size)
    total_hits = es_result.get("total_hits")
    docs = es_result["docs"]

    by_blob: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    skipped = 0
    for hit in docs:
        blob = _logical_blob_from_hit(hit)
        if not blob:
            skipped += 1
            continue
        doc_id = str(hit.get("_id", ""))
        by_blob[blob].append((doc_id, hit))

    dup_blobs = {b: rows for b, rows in by_blob.items() if len(rows) > 1}
    dup_row_count = sum(len(rows) for rows in dup_blobs.values())
    with_blob = len(docs) - skipped
    extra_rows = with_blob - len(by_blob)

    print("Query (same as CridSource.query_ids)")
    print(f"  sensor: {sensor_name}")
    print(f"  start:  {start_dt}")
    print(f"  end:    {end_dt}")
    print(f"  max_size: {max_size}")
    print(f"ES total_hits (index): {total_hits}")
    print(f"Retrieved docs: {len(docs)}")
    print(f"Docs without derivable blob: {skipped}")
    print(f"Unique logical blobs: {len(by_blob)}")
    print(f"Extra rows vs unique blobs (duplicate rows): {extra_rows}")
    print(f"Blobs with >1 document: {len(dup_blobs)}")
    print(f"Rows involved in duplicate blobs: {dup_row_count}")
    print()

    if not dup_blobs:
        print("No duplicate logical blob paths in this result set.")
        return

    for blob in sorted(dup_blobs.keys()):
        rows = dup_blobs[blob]
        print(f"--- blob ({len(rows)} docs) ---")
        print(blob)
        for doc_id, hit in rows:
            src = hit.get("_source") or {}
            ts = src.get("ts") or src.get("minute_id") or ""
            print(f"  _id={doc_id!r}  ts/minute_id={ts!r}")
        print()


if __name__ == "__main__":
    main()
