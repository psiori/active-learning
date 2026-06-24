"""Find images with anomalous features (low norm, low mean, low std).

Generates mosaics of outlier images for manual inspection.

Example:
    python3 -u find_bad_images.py
    python3 -u find_bad_images.py --norm-threshold 50 --n-show 30
"""

import argparse
import os
import sys

for k, v in list(os.environ.items()):
    if k.startswith("ENV_"):
        os.environ[k[4:]] = v
sys.path.insert(0, "/crid")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Find images with anomalous features")
    parser.add_argument(
        "--cache-file",
        default="/crid/jupyterhub/.al_feature_cache/features_cache.npz",
        help="Path to features cache .npz",
    )
    parser.add_argument(
        "--norm-threshold",
        type=float,
        default=35,
        help="Feature norm threshold for 'bad' images (default: 35)",
    )
    parser.add_argument(
        "--n-show",
        type=int,
        default=20,
        help="Number of lowest mean/std images to show (default: 20)",
    )
    parser.add_argument(
        "-s",
        "--sensor",
        default="CAM_TROLLEY",
        help="Sensor name (default: CAM_TROLLEY)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="/crid/jupyterhub/robin",
        help="Directory for output mosaics",
    )
    args = parser.parse_args()

    from datetime import datetime
    from urllib import parse
    import numpy as np
    from interface.crid import CRID
    from interface.model.models import CRIDFilter, SensorName, Dataset

    sensor = getattr(SensorName, args.sensor)

    crid = CRID(globals())

    data = np.load(args.cache_file, allow_pickle=True)
    features = data["features"]
    keys = list(data["keys"])
    norms = np.linalg.norm(features, axis=1)
    means = features.mean(axis=1)
    stds = features.std(axis=1)

    print(f"Feature stats for low-norm images (< {args.norm_threshold:.0f}):")
    low_mask = norms < args.norm_threshold
    for i in np.where(low_mask)[0]:
        print(
            f"  {keys[i]}  norm={norms[i]:.1f}  mean={means[i]:.3f}  std={stds[i]:.3f}"
        )

    print("\nLowest mean activation images:")
    sorted_by_mean = np.argsort(means)
    for i in sorted_by_mean[: args.n_show]:
        print(
            f"  {keys[i]}  norm={norms[i]:.1f}  mean={means[i]:.3f}  std={stds[i]:.3f}"
        )

    print("\nLowest std images:")
    sorted_by_std = np.argsort(stds)
    for i in sorted_by_std[: args.n_show]:
        print(
            f"  {keys[i]}  norm={norms[i]:.1f}  mean={means[i]:.3f}  std={stds[i]:.3f}"
        )

    # Generate mosaics
    f = CRIDFilter()
    f.sensor_name = sensor
    f.start_datetime = datetime(2020, 1, 1)
    f.end_datetime = datetime(2027, 1, 1)
    ds = crid.query(f, max_size=100000)

    def blob_to_fn(blob):
        items = parse.urlsplit(blob).path.split("/")[2:]
        return "_".join(items[:-4] + items[-1:])

    # Mosaic: all below norm threshold
    low_keys = set(np.array(keys)[low_mask])
    mask = ds.dataframe["blob"].map(lambda b: blob_to_fn(b) in low_keys)
    subset = Dataset()
    subset.dataframe = ds.dataframe[mask].reset_index(drop=True)
    n = len(subset.dataframe)
    print(f"\nAll norm < {args.norm_threshold:.0f}: {n} images")

    path = os.path.join(args.output_dir, "mosaic_low_norm.jpg")
    rows = (n + 9) // 10
    crid.create_image_mosaic(
        subset, path, resize_height=200, max_images=n, rows=rows, cols=min(10, n)
    )
    print("Saved", path)

    # Mosaic: lowest-mean images
    low_mean_keys = set(np.array(keys)[sorted_by_mean[: args.n_show]])
    mask2 = ds.dataframe["blob"].map(lambda b: blob_to_fn(b) in low_mean_keys)
    subset2 = Dataset()
    subset2.dataframe = ds.dataframe[mask2].reset_index(drop=True)
    path2 = os.path.join(args.output_dir, "mosaic_lowest_mean.jpg")
    crid.create_image_mosaic(
        subset2, path2, resize_height=200, max_images=args.n_show, rows=2, cols=10
    )
    print("Saved", path2)


if __name__ == "__main__":
    main()
