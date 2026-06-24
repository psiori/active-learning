"""Extract and cache ResNet50 features for all available images.

Downloads images in batches, extracts features, deletes downloaded
images to conserve disk. Features are cached persistently.

Example:
    python3 -u extract_features.py
    python3 -u extract_features.py --sensor CAM_BOOM --batch-size 500
"""

import argparse
import os
import sys
import shutil
import glob
import time

import numpy as np

for k, v in list(os.environ.items()):
    if k.startswith("ENV_"):
        os.environ[k[4:]] = v
sys.path.insert(0, "/crid")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cache_key(path: str) -> str:
    return os.path.basename(path)


def _cache_file(cache_dir: str) -> str:
    return os.path.join(cache_dir, "features_cache.npz")


def _load_cache(cache_dir: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    cache_file = _cache_file(cache_dir)
    if not os.path.exists(cache_file):
        return (
            [],
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
        )
    data = np.load(cache_file, allow_pickle=True)
    keys = list(data["keys"])
    features = data["features"]
    stats = (
        data["stats"] if "stats" in data else np.empty((len(keys), 0), dtype=np.float32)
    )
    return keys, features, stats


def load_cached_features(
    cache_dir: str,
    image_paths: list[str],
) -> tuple[np.ndarray | None, list[str], list[int]]:
    keys, features, _stats = _load_cache(cache_dir)
    if not keys:
        return None, list(image_paths), list(range(len(image_paths)))

    key_to_idx = {k: i for i, k in enumerate(keys)}
    cached_rows = []
    uncached_paths = []
    uncached_indices = []
    for i, path in enumerate(image_paths):
        key = _cache_key(path)
        if key in key_to_idx:
            cached_rows.append(features[key_to_idx[key]])
        else:
            uncached_paths.append(path)
            uncached_indices.append(i)

    cached = np.asarray(cached_rows) if cached_rows else None
    return cached, uncached_paths, uncached_indices


def save_features(
    cache_dir: str,
    image_paths: list[str],
    features: np.ndarray,
    stats: np.ndarray | None = None,
) -> None:
    cache_file = _cache_file(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    existing_keys, existing_features, existing_stats = _load_cache(cache_dir)
    key_to_idx = {k: i for i, k in enumerate(existing_keys)}

    new_keys = [_cache_key(p) for p in image_paths]
    if existing_features.size == 0:
        all_keys = list(new_keys)
        all_features = np.asarray(features)
        all_stats = (
            np.asarray(stats)
            if stats is not None
            else np.empty((len(new_keys), 0), dtype=np.float32)
        )
    else:
        all_keys = list(existing_keys)
        all_features = existing_features
        all_stats = existing_stats
        for i, key in enumerate(new_keys):
            if key in key_to_idx:
                idx = key_to_idx[key]
                all_features[idx] = features[i]
                if stats is not None and all_stats.size:
                    all_stats[idx] = stats[i]
            else:
                all_keys.append(key)
                all_features = np.concatenate(
                    [all_features, features[i : i + 1]],
                    axis=0,
                )
                if stats is not None:
                    if all_stats.size == 0:
                        all_stats = stats[i : i + 1]
                    else:
                        all_stats = np.concatenate(
                            [all_stats, stats[i : i + 1]],
                            axis=0,
                        )
                elif all_stats.size:
                    all_stats = np.concatenate(
                        [
                            all_stats,
                            np.full(
                                (1, all_stats.shape[1]),
                                np.nan,
                                dtype=all_stats.dtype,
                            ),
                        ],
                        axis=0,
                    )

    save_dict = {"keys": np.array(all_keys), "features": all_features}
    if stats is not None:
        save_dict["stats"] = all_stats
    elif existing_stats.size:
        save_dict["stats"] = existing_stats
    np.savez(cache_file, **save_dict)


def main():
    parser = argparse.ArgumentParser(description="Extract and cache ResNet50 features")
    parser.add_argument(
        "-s",
        "--sensor",
        default="CAM_TROLLEY",
        help="Sensor name (default: CAM_TROLLEY)",
    )
    parser.add_argument(
        "--cache-dir",
        default="/crid/jupyterhub/.al_feature_cache",
        help="Cache directory (default: /crid/jupyterhub/.al_feature_cache)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Download batch size (default: 1000)",
    )
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=32,
        help="Feature extraction batch size (default: 32)",
    )
    parser.add_argument(
        "--temp-dir",
        default="/tmp/al_feature_extract",
        help="Temp directory for downloads (default: /tmp/al_feature_extract)",
    )
    args = parser.parse_args()

    from datetime import datetime
    from interface.crid import CRID
    from interface.model.models import CRIDFilter, SensorName, Dataset
    from active_learning.scorers.features import get_default_extractor, extract_features

    sensor = getattr(SensorName, args.sensor)

    print("Initializing CRID...")
    crid = CRID(globals())

    print(f"Querying all {args.sensor} images...")
    crid_filter = CRIDFilter()
    crid_filter.sensor_name = sensor
    crid_filter.start_datetime = datetime(2020, 1, 1)
    crid_filter.end_datetime = datetime(2027, 1, 1)
    all_dataset = crid.query(crid_filter, max_size=100000)
    total = len(all_dataset.dataframe)
    print(f"Total images found: {total}")

    print("Loading ResNet50 model...")
    extractor = get_default_extractor()

    os.makedirs(args.cache_dir, exist_ok=True)

    df = all_dataset.dataframe
    n_batches = (total + args.batch_size - 1) // args.batch_size
    processed = 0
    cached_count = 0
    extracted_count = 0
    start_time = time.time()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total)
        batch_df = df.iloc[batch_start:batch_end]

        print(
            f"\n--- Batch {batch_idx + 1}/{n_batches} (images {batch_start}-{batch_end - 1}) ---",
        )

        batch_dir = os.path.join(args.temp_dir, f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)

        batch_dataset = Dataset()
        batch_dataset.dataframe = batch_df.reset_index(drop=True)
        print(f"Downloading {len(batch_df)} images...")
        crid.download_dataset(batch_dataset, destination_dir=batch_dir, strict=False)

        image_paths = sorted(
            glob.glob(os.path.join(batch_dir, "*.png"))
            + glob.glob(os.path.join(batch_dir, "*.jpg")),
        )
        image_paths = [p for p in image_paths if os.path.getsize(p) > 0]

        if not image_paths:
            print("No images downloaded in this batch, skipping")
            shutil.rmtree(batch_dir, ignore_errors=True)
            continue

        print(f"Downloaded {len(image_paths)} images")

        cached, uncached_paths, uncached_indices = load_cached_features(
            args.cache_dir,
            image_paths,
        )

        if not uncached_paths:
            print(f"All {len(image_paths)} already cached, skipping")
            cached_count += len(image_paths)
        else:
            print(
                f"Cached: {len(image_paths) - len(uncached_paths)}, extracting: {len(uncached_paths)}",
            )
            cached_count += len(image_paths) - len(uncached_paths)

            new_features, new_stats = extract_features(
                uncached_paths,
                extractor=extractor,
                batch_size=args.feature_batch_size,
            )
            save_features(args.cache_dir, uncached_paths, new_features, new_stats)
            extracted_count += len(uncached_paths)

        processed += len(image_paths)
        shutil.rmtree(batch_dir, ignore_errors=True)

        elapsed = time.time() - start_time
        rate = processed / elapsed
        remaining = (total - batch_end) / rate if rate > 0 else 0
        print(
            f"Progress: {processed}/{total}, {extracted_count} extracted, {cached_count} cached, "
            f"~{remaining / 60:.0f}min remaining",
        )

    shutil.rmtree(args.temp_dir, ignore_errors=True)

    elapsed = time.time() - start_time
    cache_file = _cache_file(args.cache_dir)
    cache_size = os.path.getsize(cache_file) if os.path.exists(cache_file) else 0
    print(f"\nDone! {processed} images in {elapsed / 60:.1f} min")
    print(f"  Extracted: {extracted_count}, From cache: {cached_count}")
    print(f"  Cache size: {cache_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
