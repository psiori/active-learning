"""Compute uncertainty and run uncertainty-aware selection."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from active_learning.integrations.crid.provider_source import CridImageProviderSource
from active_learning.integrations.crid.source import CridSource
from active_learning.core import (
    build_image_provider,
    run_uncertainty_selection,
)
from active_learning.providers import enable_mc_dropout, load_unet
from active_learning.sinks.mosaic import render_mosaic
from interface.crid import CRID
from interface.model.models import SensorName


for key, value in list(os.environ.items()):
    if key.startswith("ENV_"):
        os.environ[key[4:]] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute uncertainty and run selection",
    )
    parser.add_argument("-n", "--n-select", type=int, default=200)
    parser.add_argument("-a", "--alpha", type=float, default=0.7)
    parser.add_argument("-m", "--model-name", default="geiles_unet_250912")
    parser.add_argument(
        "--model-path",
        default="/crid/jupyterhub/.models/geiles_unet_250912_0000000_custom_fold1.zip",
    )
    parser.add_argument("-s", "--sensor", default="CAM_TROLLEY")
    parser.add_argument(
        "--provider",
        default="mc_dropout",
        choices=["mc_dropout", "entropy", "bald"],
    )
    parser.add_argument("--mc-iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--aggregation",
        default="topk_mean",
        choices=["mean", "topk_mean", "max"],
    )
    parser.add_argument("--topk-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-o",
        "--output",
        default="/crid/jupyterhub/robin/mosaic_uncertainty.jpg",
    )
    parser.add_argument("--sama-project-id", type=int, default=None)
    parser.add_argument("--no-sama", action="store_true")
    parser.add_argument("--cache-root", default="/crid/jupyterhub/.al_feature_cache")
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

    print(f"Computing {args.provider} uncertainty...")
    start = time.time()
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
        alpha=args.alpha,
        iterations=args.mc_iterations,
        batch_size=args.batch_size,
        aggregation=args.aggregation,
        topk_fraction=args.topk_fraction,
        strategy="uncertainty_coreset",
        seed=args.seed,
    )
    elapsed = time.time() - start
    print(f"Selected {len(result.selected_ids)} in {elapsed:.1f}s")

    unc_scores = np.array(list(result.scores.values()), dtype=np.float32)
    if len(unc_scores) > 0:
        print(
            "Uncertainty distribution:",
            f"min={unc_scores.min():.6f}",
            f"max={unc_scores.max():.6f}",
            f"mean={unc_scores.mean():.6f}",
            f"std={unc_scores.std():.6f}",
        )

    render_mosaic(
        result,
        image_provider,
        args.output,
        resize_height=150,
        max_images=200,
        rows=10,
        cols=20,
    )
    print("Saved", args.output)


if __name__ == "__main__":
    main()
