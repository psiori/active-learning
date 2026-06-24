"""Brightness bounds in YAML may be null; parsing must stay numeric-safe."""

from active_learning.core.config import QueryConfig, parse_query


def test_parse_query_null_brightness_drops_to_defaults():
    cfg = parse_query(
        {
            "query": {
                "sensor": "CAM_TROLLEY",
                "min_brightness": None,
                "max_brightness": None,
            },
        },
    )
    assert isinstance(cfg, QueryConfig)
    assert cfg.min_brightness == 0.0
    assert cfg.max_brightness == 220.0


def test_parse_query_partial_null_brightness():
    cfg = parse_query(
        {
            "query": {
                "sensor": "CAM_TROLLEY",
                "min_brightness": None,
                "max_brightness": 180.0,
            },
        },
    )
    assert cfg.min_brightness == 0.0
    assert cfg.max_brightness == 180.0


def test_parse_query_brightness_filter_disabled_bool():
    cfg = parse_query(
        {
            "query": {
                "sensor": "CAM_TROLLEY",
                "brightness_filter_enabled": False,
            },
        },
    )
    assert cfg.brightness_filter_enabled is False
