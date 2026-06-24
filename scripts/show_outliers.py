"""Show images with extreme feature norms (likely blank/overexposed).

Example:
    python3 -u show_outliers.py
    python3 -u show_outliers.py --n-show 10
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
    parser = argparse.ArgumentParser(
        description="Show images with extreme feature norms"
    )
    parser.add_argument(
        "--n-show",
        type=int,
        default=5,
        help="Number of outliers from each end (default: 5)",
    )
    parser.add_argument(
        "--cache-file",
        default="/crid/jupyterhub/.al_feature_cache/features_cache.npz",
        help="Path to features cache .npz",
    )
    parser.add_argument(
        "-s",
        "--sensor",
        default="CAM_TROLLEY",
        help="Sensor name (default: CAM_TROLLEY)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/crid/jupyterhub/robin/mosaic_outliers.jpg",
        help="Mosaic output path",
    )
    args = parser.parse_args()

    import numpy as np
    from datetime import datetime
    from urllib import parse
    from interface.crid import CRID
    from interface.model.models import CRIDFilter, SensorName, Dataset

    sensor = getattr(SensorName, args.sensor)

    crid = CRID(globals())

    data = np.load(args.cache_file, allow_pickle=True)
    features = data["features"]
    keys = list(data["keys"])
    norms = np.linalg.norm(features, axis=1)
    sorted_idx = np.argsort(norms)

    outlier_names = [keys[i] for i in sorted_idx[: args.n_show]] + [
        keys[i] for i in sorted_idx[-args.n_show :]
    ]
    print("Lowest norm:")
    for i in sorted_idx[: args.n_show]:
        print(f"  {keys[i]}  norm={norms[i]:.1f}")
    print("Highest norm:")
    for i in sorted_idx[-args.n_show :]:
        print(f"  {keys[i]}  norm={norms[i]:.1f}")

    f = CRIDFilter()
    f.sensor_name = sensor
    f.start_datetime = datetime(2020, 1, 1)
    f.end_datetime = datetime(2027, 1, 1)
    ds = crid.query(f, max_size=100000)

    def blob_to_fn(blob):
        items = parse.urlsplit(blob).path.split("/")[2:]
        return "_".join(items[:-4] + items[-1:])

    outlier_set = set(outlier_names)
    mask = ds.dataframe["blob"].map(lambda b: blob_to_fn(b) in outlier_set)
    outlier_ds = Dataset()
    outlier_ds.dataframe = ds.dataframe[mask].reset_index(drop=True)
    print(f"Found {len(outlier_ds.dataframe)} rows")

    total = args.n_show * 2
    rows = max(1, (total + 9) // 10)
    cols = min(10, total)
    crid.create_image_mosaic(
        outlier_ds,
        args.output,
        resize_height=200,
        max_images=total,
        rows=rows,
        cols=cols,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
