"""UNet loading and model helpers for active learning."""

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import tensorflow as tf

DEFAULT_PSIPY_PATH = "/crid/jupyterhub/.psipy"


def load_unet(model_path: str, psipy_path: str = DEFAULT_PSIPY_PATH):
    """Load a psipy UNet model from a .zip file."""
    unet_cls = _load_unet_class(psipy_path)
    resolved_model_path = _resolve_model_path(model_path)
    unet = unet_cls.load(resolved_model_path)
    try:
        unet._model_path = resolved_model_path
    except Exception:
        pass
    return unet


def unet_cache_namespace(
    unet,
    model_name: str | None = None,
    model_path: str | None = None,
) -> str:
    """Build a stable cache namespace for a UNet instance."""
    try:
        config = unet.get_config()
    except Exception:
        config = {"model_class": type(unet).__name__}

    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    path = model_path or getattr(unet, "_model_path", None)
    if path and os.path.exists(path):
        digest = f"{digest}-{_file_sha1(path)}"
    prefix = model_name or "unet"
    return f"{prefix}_{digest}"


def enable_mc_dropout(unet):
    """Rebuild a UNet with inference dropout enabled."""
    unet_cls = _load_unet_class(DEFAULT_PSIPY_PATH)
    configs = dict(unet.get_config())
    configs["inference_dropout"] = True
    mc_unet = unet_cls.from_config(configs)
    mc_unet.model.set_weights(unet.model.get_weights())
    return mc_unet


def create_penultimate_model(unet):
    """Create a TF model that outputs probs and penultimate activations."""
    model = unet.model
    probs_output = model.get_layer("probs").output
    penultimate_output = model.get_layer("logits").input

    return tf.keras.Model(
        inputs=model.input,
        outputs=[probs_output, penultimate_output],
    )


def _load_unet_class(psipy_path: str):
    if psipy_path not in sys.path:
        sys.path.insert(0, psipy_path)
    return importlib.import_module("psipy.vision.model").UNet


def _resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if not path.is_dir():
        return str(path)

    sibling_zip = path.with_suffix(".zip")
    if sibling_zip.is_file():
        return str(sibling_zip)

    zip_files = sorted(path.glob("*.zip"))
    if len(zip_files) == 1:
        return str(zip_files[0])

    if zip_files:
        options = ", ".join(str(p) for p in zip_files)
        raise IsADirectoryError(
            f"UNet model_path {model_path!r} is a directory with multiple .zip "
            f"files. Configure one of: {options}"
        )

    raise IsADirectoryError(
        f"UNet model_path {model_path!r} is a directory. Configure the .zip file "
        f"directly or place it at {str(sibling_zip)!r}."
    )


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]
