# CRID Integration

CRID-specific source and export helpers.

- `source.py`: queries pool and labeled IDs from CRID.
- `provider_source.py`: downloads CRID images for the shared `ImageProvider`.
- `export.py`: builds export IDs and descriptions and writes selected IDs back to CRID.

This package is the main bridge between active-learning logic and CRID runtime services.
