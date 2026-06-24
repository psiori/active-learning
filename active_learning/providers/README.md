# Providers

Model-facing inference and uncertainty backends.

- `unet.py`: model loading, caching, and MC-dropout setup.
- `inference.py`: batched image inference utilities.
- `entropy.py`, `bald.py`, `mc_dropout.py`: uncertainty provider implementations.
- `aggregation.py`: pixel-map to scalar aggregation helpers.

Use this package for "how do we get model outputs?" questions.
