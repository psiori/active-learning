"""Provider submodule exports."""

from active_learning.providers.aggregation import (
    aggregate_uncertainty_map,
    entropy_map,
)
from active_learning.providers.bald import bald_provider, compute_bald_uncertainty
from active_learning.providers.entropy import (
    compute_entropy_uncertainty,
    entropy_provider,
)
from active_learning.providers.inference import (
    build_infer_fn,
    collect_mc_probs,
    extract_probs,
    extract_probs_and_penultimate,
    iter_image_batches,
)
from active_learning.providers.mc_dropout import (
    compute_mc_uncertainty,
    mc_dropout_provider,
)
from active_learning.providers.unet import (
    create_penultimate_model,
    enable_mc_dropout,
    load_unet,
    unet_cache_namespace,
)

__all__ = [
    "aggregate_uncertainty_map",
    "bald_provider",
    "build_infer_fn",
    "collect_mc_probs",
    "compute_bald_uncertainty",
    "compute_entropy_uncertainty",
    "compute_mc_uncertainty",
    "create_penultimate_model",
    "enable_mc_dropout",
    "entropy_map",
    "entropy_provider",
    "extract_probs",
    "extract_probs_and_penultimate",
    "iter_image_batches",
    "load_unet",
    "mc_dropout_provider",
    "unet_cache_namespace",
]
