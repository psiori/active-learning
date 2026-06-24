"""TensorFlow logging and runtime helpers."""

from __future__ import annotations

import os


def ensure_tensorflow_log_suppression() -> None:
    """Set TensorFlow C++ log suppression before TensorFlow is imported."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def configure_tensorflow_logging() -> None:
    """Silence TensorFlow and absl Python-side logs."""
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass


def describe_tensorflow_device() -> str:
    """Return a one-line summary of the TensorFlow runtime device."""
    import tensorflow as tf

    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        device = gpu_devices[0]
        details = {}
        try:
            details = tf.config.experimental.get_device_details(device) or {}
        except Exception:
            details = {}
        device_name = (
            details.get("device_name")
            or details.get("name")
            or getattr(device, "name", "GPU")
        )
        device_type = getattr(device, "device_type", "GPU")
        return f"TensorFlow device: {device_type} ({device_name})"

    cpu_devices = tf.config.list_physical_devices("CPU")
    if cpu_devices:
        device = cpu_devices[0]
        device_name = getattr(device, "name", "CPU")
        device_type = getattr(device, "device_type", "CPU")
        return f"TensorFlow device: {device_type} ({device_name})"

    return "TensorFlow device: unavailable"
