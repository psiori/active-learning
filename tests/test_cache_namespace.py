"""Tests for cache namespacing helpers."""

from active_learning.providers.unet import unet_cache_namespace


class _DummyUnet:
    def __init__(self, config):
        self._config = config

    def get_config(self):
        return self._config


def test_unet_cache_namespace_changes_with_config():
    a = _DummyUnet({"input_shape": (320, 240, 3), "num_classes": 4})
    b = _DummyUnet({"input_shape": (320, 240, 3), "num_classes": 8})
    ns_a = unet_cache_namespace(a, "poeppelmann_only")
    ns_b = unet_cache_namespace(b, "poeppelmann_only")
    assert ns_a != ns_b
    assert ns_a.startswith("poeppelmann_only_")
    assert ns_b.startswith("poeppelmann_only_")
