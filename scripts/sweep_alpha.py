"""Run uncertainty-weighted selection at different alpha values."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from active_learning.integrations.crid.provider_source import CridImageProviderSource
from active_learning.integrations.crid.source import CridSource
from active_learning.core import build_image_provider, run_uncertainty_selection
from active_learning.providers import enable_mc_dropout, load_unet
from active_learning.sinks.mosaic import render_mosaic
from interface.crid import CRID
from interface.model.models import SensorName


for key, value in list(os.environ.items()):
    if key.startswith("ENV_"):
        os.environ[key[4:]] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep alpha for uncertainty-weighted coreset",
    )
    parser.add_argument("-n", "--n-select", type=int, default=200)
    parser.add_argument("-m", "--model-name", default="geiles_unet_250912")
    parser.add_argument("-s", "--sensor", default="CAM_TROLLEY")
    parser.add_argument("--cache-root", default="/crid/jupyterhub/.al_feature_cache")
    parser.add_argument(
        "--provider",
        default="mc_dropout",
        choices=["mc_dropout", "entropy", "bald"],
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output-dir", default="/crid/jupyterhub/robin")
    parser.add_argument("--sama-project-id", type=int, default=None)
    parser.add_argument("--no-sama", action="store_true")
    parser.add_argument(
        "--model-path",
        default="/crid/jupyterhub/.models/geiles_unet_250912_0000000_custom_fold1.zip",
    )
    parser.add_argument("--image-size", nargs=2, type=int, default=[320, 240])
    parser.add_argument("--mc-iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--aggregation",
        default="topk_mean",
        choices=["mean", "topk_mean", "max"],
    )
    parser.add_argument("--topk-fraction", type=float, default=0.10)
    args = parser.parse_args()

    crid = CRID(globals())
    source = CridSource(crid)
    provider_source = CridImageProviderSource(
        crid,
        Path(args.cache_root) / "downloads",
    )
    image_provider = build_image_provider(provider_source, args.cache_root)
    queried = source.query_pool_and_labeled_ids(
        sensor_name=getattr(SensorName, args.sensor),
        sama_project_id=args.sama_project_id,
        exclude_seeded=not args.no_sama,
    )
    unet = load_unet(args.model_path)
    uncertainty_model = (
        enable_mc_dropout(unet) if args.provider in {"mc_dropout", "bald"} else unet
    )

    for alpha in args.alphas:
        strategy = "uncertainty_coreset" if alpha < 1.0 else "uncertainty_coreset"
        result = run_uncertainty_selection(
            queried.pool_ids,
            seed_ids=queried.labeled_ids,
            image_provider=image_provider,
            cache_root=args.cache_root,
            n=args.n_select,
            uncertainty_kind=args.provider,
            uncertainty_model=uncertainty_model,
            feature_model="resnet50",
            model_name=args.model_name,
            model_path=args.model_path,
            alpha=alpha,
            iterations=args.mc_iterations,
            image_size=tuple(args.image_size),
            batch_size=args.batch_size,
            aggregation=args.aggregation,
            topk_fraction=args.topk_fraction,
            strategy=strategy,
            seed=args.seed,
        )
        output = Path(args.output_dir) / f"mosaic_alpha_{alpha:.1f}.jpg"
        render_mosaic(
            result,
            image_provider,
            output,
            resize_height=150,
            max_images=200,
            rows=10,
            cols=20,
        )
        print(f"alpha={alpha:.1f}: saved {output}")


if __name__ == "__main__":
    main()
