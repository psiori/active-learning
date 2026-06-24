# Scorers

Feature extraction, uncertainty scoring, and lightweight pre-selection helpers.

- `brightness.py`: scalar brightness scoring and brightness-based filtering.
- `features.py`: feature embedding extraction.
- `uncertainty.py`: uncertainty scoring entrypoints.
- `alges.py`: ALGES-specific embedding/scoring helpers.
- `_cache.py`: cache support for expensive derived artifacts.

Most scorers compute values; brightness also exposes a small filtering helper built on those cached scores.
