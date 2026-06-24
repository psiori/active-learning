# Active Learning Scripts

This directory contains:

- active-learning entrypoints built on the current `active_learning` package
- small payload-inspection tools
- a few lower-level CRID utility scripts for stats and feature exploration

Most scripts are intended to run in a CRID environment because they import
`interface.crid.CRid` and access project-specific models, Azure-backed image
data, or Sama credentials.

## Main Entry Points

### `seed.py`

Select images from CRID using one strategy, render a mosaic, write an
interactive seed payload, and optionally export the selection plus create a
Sama batch.

Current strategies:

- `coreset`
- `uncertainty_coreset`
- `uncertainty_topk`
- `uncertainty_topk_coreset`
- `alges`
- `alges_coreset`

Important flags:

- `-c, --config`
- `-p, --project`
- `--strategy`
- `-n, --n-select`
- `--rng-seed`
- `-s, --sensor`
- `--cache-root`
- `--start`
- `--end`
- `--sama-project-id`
- `--sama-priority`
- `--exclude-seeded`
- `--min-brightness`
- `--max-brightness`
- `--use-full-res-images`
- `--feature-model`
- `-a, --alpha`
- `--provider {mc_dropout,entropy,bald}`
- `--mc-iterations`
- `--batch-size`
- `--aggregation {mean,topk_mean,max}`
- `--topk-fraction`
- `--candidate-multiplier`
- `--method {image,semantic}`
- `--export-prefix`
- `--mosaic-path`
- `--export-sama`
- `--overlay`
- `--model-path`
- `--model-name`

Notes:

- seeding/export is opt-in and only happens when `--export-sama` is set
- brightness filtering is applied before selection
- output includes:
  - a mosaic image
  - an interactive seed YAML payload
  - optional CRID export
  - optional Sama batch creation

Example:

```bash
python3 -u ../seed.py \
  --config ../al_config.yaml \
  --project skasoegel \
  --strategy uncertainty_topk_coreset \
  --alpha 0.5 \
  --provider bald \
  --candidate-multiplier 4
```

### `compare_strategies.py`

Run multiple strategies on the same queried pool and save one mosaic per
strategy.

Supported strategies:

- `coreset`
- `uncertainty_coreset`
- `uncertainty_topk`
- `uncertainty_topk_coreset`
- `alges`
- `alges_coreset`

Important flags:

- `-c, --config`
- `-p, --project`
- `--strategies`
- `-o, --output-dir`
- all major selection/query/model flags from the main `seed.py`

Example:

```bash
python3 -u compare_strategies.py \
  --config ../al_config.yaml \
  --project skasoegel \
  --strategies coreset uncertainty_topk alges
```

### `compute_uncertainty.py`

Compute uncertainty over a CRID pool and run uncertainty-aware selection.

Important flags:

- `-n, --n-select`
- `-a, --alpha`
- `-m, --model-name`
- `--model-path`
- `-s, --sensor`
- `--provider {mc_dropout,entropy,bald}`
- `--mc-iterations`
- `--batch-size`
- `--aggregation {mean,topk_mean,max}`
- `--topk-fraction`
- `--seed`
- `-o, --output`
- `--sama-project-id`
- `--no-sama`
- `--cache-root`

Example:

```bash
python3 -u compute_uncertainty.py \
  --provider bald \
  --mc-iterations 10 \
  --n-select 100
```

### `sweep_alpha.py`

Run `uncertainty_coreset` selection for multiple alpha values and save one
mosaic per alpha.

Important flags:

- `--alphas`
- `-n, --n-select`
- `-m, --model-name`
- `--model-path`
- `-s, --sensor`
- `--provider {mc_dropout,entropy,bald}`
- `--mc-iterations`
- `--batch-size`
- `--aggregation {mean,topk_mean,max}`
- `--topk-fraction`
- `--cache-root`
- `-o, --output-dir`
- `--sama-project-id`
- `--no-sama`
- `--seed`

Notes:

- the current script sweeps alpha within `uncertainty_coreset`
- output files are named `mosaic_alpha_<value>.jpg`

Example:

```bash
python3 -u sweep_alpha.py \
  --alphas 0.2 0.5 0.8 \
  --provider mc_dropout \
  --n-select 100
```

## Utility Scripts

These scripts are more task-specific and lower-level than the main entrypoints:

- `compute_stats.py`
  - compute pixel stats and generate threshold mosaics
- `extract_features.py`
  - extract and cache ResNet50 features
- `find_bad_images.py`
  - find anomalous images using cached features
- `show_dark.py`
  - inspect darkest images from cached stats
- `show_outliers.py`
  - inspect feature outliers
- `overlay_helpers.py`
  - helper functions for rendering mosaics from selected IDs and an `ImageProvider`

## Configuration

The main config file used by the active-learning entrypoints is:

- [al_config.yaml](/Users/lukas/repos/active-learning/active_learning/al_config.yaml)

Config precedence:

- CLI arguments
- YAML config
- dataclass defaults

## Environment Notes

- `ENV_*` variables are mirrored into unprefixed environment variables by the
  main active-learning entrypoints
- TensorFlow and model artifacts must be available in the execution environment
- CRID-backed scripts require access to the CRID runtime and project-specific
  credentials/configuration
