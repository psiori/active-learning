"""Compute pixel stats for all cached images and generate threshold mosaics.

Example:
    python3 -u compute_stats.py
    python3 -u compute_stats.py --sensor CAM_BOOM --batch-size 500
"""

import argparse
import os
import sys
import shutil
import time

for k, v in list(os.environ.items()):
    if k.startswith("ENV_"):
        os.environ[k[4:]] = v
sys.path.insert(0, "/crid")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Compute pixel stats and generate threshold mosaics",
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
        "-s",
        "--sensor",
        default="CAM_TROLLEY",
        help="Sensor name (default: CAM_TROLLEY)",
    )
    parser.add_argument(
        "--temp-dir",
        default="/tmp/al_stats_compute",
        help="Temp directory for downloads",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="/crid/jupyterhub/robin",
        help="Directory for output mosaics",
    )
    args = parser.parse_args()

    import numpy as np
    from datetime import datetime
    from urllib import parse
    from interface.crid import CRID
    from interface.model.models import CRIDFilter, SensorName, Dataset
    from active_learning.scorers.features import compute_image_stats, STAT_KEYS

    sensor = getattr(SensorName, args.sensor)
    cache_file = os.path.join(args.cache_dir, "features_cache.npz")

    print("Loading cache...")
    data = np.load(cache_file, allow_pickle=True)
    keys = list(data["keys"])
    features = data["features"]
    n_total = len(keys)
    print(f"Cache has {n_total} entries")

    # Check if stats already exist
    if "stats" in data and not np.all(np.isnan(data["stats"])):
        existing_stats = data["stats"]
        n_with_stats = np.sum(~np.isnan(existing_stats[:, 0]))
        print(f"Already have stats for {n_with_stats} images")
    else:
        existing_stats = np.full((n_total, len(STAT_KEYS)), np.nan, dtype=np.float32)
        n_with_stats = 0

    need_stats = np.where(np.isnan(existing_stats[:, 0]))[0]
    print(f"Need stats for {len(need_stats)} images")

    if len(need_stats) == 0:
        print("All stats computed!")
    else:
        # Init CRID for downloading
        print("Initializing CRID...")
        crid = CRID(globals())

        f = CRIDFilter()
        f.sensor_name = sensor
        f.start_datetime = datetime(2020, 1, 1)
        f.end_datetime = datetime(2027, 1, 1)
        ds = crid.query(f, max_size=1000000)

        def blob_to_fn(blob):
            items = parse.urlsplit(blob).path.split("/")[2:]
            return "_".join(items[:-4] + items[-1:])

        # Build filename -> blob mapping
        fn_to_blob = {}
        for blob in ds.dataframe["blob"]:
            fn_to_blob[blob_to_fn(blob)] = blob

        # Process in batches
        keys_needing_stats = [keys[i] for i in need_stats]
        n_batches = (len(keys_needing_stats) + args.batch_size - 1) // args.batch_size
        start_time = time.time()
        computed = 0

        for batch_idx in range(n_batches):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(keys_needing_stats))
            batch_keys = keys_needing_stats[batch_start:batch_end]

            print(
                f"\n--- Batch {batch_idx + 1}/{n_batches} ({len(batch_keys)} images) ---",
            )

            # Find blobs for these keys and download
            batch_blobs = [fn_to_blob.get(k) for k in batch_keys]
            valid = [(k, b) for k, b in zip(batch_keys, batch_blobs) if b is not None]
            if not valid:
                print("No matching blobs found, skipping")
                continue

            batch_dataset = Dataset()
            batch_dataset.dataframe = ds.dataframe[
                ds.dataframe["blob"].isin([b for _, b in valid])
            ].reset_index(drop=True)

            temp_dir = os.path.join(args.temp_dir, f"batch_{batch_idx}")
            os.makedirs(temp_dir, exist_ok=True)
            crid.download_dataset(batch_dataset, destination_dir=temp_dir, strict=False)

            # Compute stats for downloaded images
            from PIL import Image

            for fn, blob in valid:
                local_path = os.path.join(temp_dir, fn)
                if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                    continue

                try:
                    img = Image.open(local_path).resize((224, 224))
                    img_array = np.array(img, dtype=np.float32)
                    if img_array.ndim == 2:
                        img_array = np.stack([img_array] * 3, axis=2)
                    stats = compute_image_stats(img_array)
                    cache_idx = keys.index(fn)
                    existing_stats[cache_idx] = [stats[k] for k in STAT_KEYS]
                    computed += 1
                except Exception as e:
                    print(f"Error processing {fn}: {e}")

            shutil.rmtree(temp_dir, ignore_errors=True)

            elapsed = time.time() - start_time
            rate = computed / elapsed if elapsed > 0 else 0
            remaining = (len(need_stats) - computed) / rate / 60 if rate > 0 else 0
            print(
                f"Computed: {computed}/{len(need_stats)}, ~{remaining:.0f} min remaining",
            )

        shutil.rmtree(args.temp_dir, ignore_errors=True)

        # Save updated cache
        print("\nSaving cache with stats...")
        np.savez(
            cache_file,
            keys=np.array(keys),
            features=features,
            stats=existing_stats,
        )
        n_with_stats = np.sum(~np.isnan(existing_stats[:, 0]))
        print(f"Done! {n_with_stats}/{n_total} images have stats")

    # --- Generate mosaics at different brightness thresholds ---
    print("\n=== Generating threshold mosaics ===")

    data = np.load(cache_file, allow_pickle=True)
    keys = list(data["keys"])
    stats = data["stats"] if "stats" in data else None

    if stats is None or np.all(np.isnan(stats[:, 0])):
        print("No stats available yet, skipping mosaics")
        sys.exit(0)

    brightness = stats[:, 0]
    brightness_std = stats[:, 1]
    valid_mask = ~np.isnan(brightness)

    print("Brightness stats (valid images):")
    valid_brightness = brightness[valid_mask]
    print(
        f"  min={valid_brightness.min():.1f} max={valid_brightness.max():.1f} "
        f"mean={valid_brightness.mean():.1f} std={valid_brightness.std():.1f}",
    )

    for p in [1, 2, 5, 95, 98, 99]:
        val = np.percentile(valid_brightness, p)
        print(f"  p{p}={val:.1f}")

    if "crid" not in dir():
        crid = CRID(globals())
    f = CRIDFilter()
    f.sensor_name = sensor
    f.start_datetime = datetime(2020, 1, 1)
    f.end_datetime = datetime(2027, 1, 1)
    ds = crid.query(f, max_size=1000000)

    def blob_to_fn(blob):
        items = parse.urlsplit(blob).path.split("/")[2:]
        return "_".join(items[:-4] + items[-1:])

    ranges = [
        ("very_dark_below_30", valid_mask & (brightness < 30)),
        ("dark_30_60", valid_mask & (brightness >= 30) & (brightness < 60)),
        ("dark_60_80", valid_mask & (brightness >= 60) & (brightness < 80)),
        ("very_bright_above_220", valid_mask & (brightness > 220)),
        ("bright_200_220", valid_mask & (brightness >= 200) & (brightness <= 220)),
        ("bright_180_200", valid_mask & (brightness >= 180) & (brightness < 200)),
        ("low_std_below_20", valid_mask & (brightness_std < 20)),
    ]

    for name, mask_arr in ranges:
        selected_keys = set(np.array(keys)[mask_arr])
        df_mask = ds.dataframe["blob"].map(
            lambda b, s=selected_keys: blob_to_fn(b) in s,
        )
        subset = Dataset()
        subset.dataframe = ds.dataframe[df_mask].reset_index(drop=True)
        n = len(subset.dataframe)
        print(f"{name}: {n} images")
        if n > 0:
            rows = min(5, (min(n, 50) + 9) // 10)
            cols = min(10, n)
            path = os.path.join(args.output_dir, f"mosaic_brightness_{name}.jpg")
            crid.create_image_mosaic(
                subset,
                path,
                resize_height=200,
                max_images=50,
                rows=rows,
                cols=cols,
            )
            print(f"  saved {path}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
