"""Show the darkest images from cached image statistics.

Example:
    python3 -u show_dark.py
    python3 -u show_dark.py --n-show 10 --sensor CAM_BOOM
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
        description="Show darkest images from cached stats"
    )
    parser.add_argument(
        "--n-show",
        type=int,
        default=4,
        help="Number of darkest images to show (default: 4)",
    )
    parser.add_argument(
        "--stats-file",
        default="/crid/jupyterhub/.al_feature_cache/image_stats.npz",
        help="Path to image stats .npz",
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
        default="/crid/jupyterhub/robin/mosaic_darkest.jpg",
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

    data = np.load(args.stats_file, allow_pickle=True)
    keys = list(data["keys"])
    stats = data["values"]
    brightness = stats[:, 0]

    dark_idx = np.argsort(brightness)[: args.n_show]
    dark_keys = set(keys[i] for i in dark_idx)
    for i in dark_idx:
        print(f"{keys[i]}  brightness={brightness[i]:.1f}  std={stats[i, 1]:.1f}")

    f = CRIDFilter()
    f.sensor_name = sensor
    f.start_datetime = datetime(2020, 1, 1)
    f.end_datetime = datetime(2027, 1, 1)
    ds = crid.query(f, max_size=100000)

    def blob_to_fn(blob):
        items = parse.urlsplit(blob).path.split("/")[2:]
        return "_".join(items[:-4] + items[-1:])

    mask = ds.dataframe["blob"].map(lambda b: blob_to_fn(b) in dark_keys)
    subset = Dataset()
    subset.dataframe = ds.dataframe[mask].reset_index(drop=True)

    rows = max(1, (args.n_show + 9) // 10)
    cols = min(10, args.n_show)
    crid.create_image_mosaic(
        subset,
        args.output,
        resize_height=300,
        max_images=args.n_show,
        rows=rows,
        cols=cols,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
