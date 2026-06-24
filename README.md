# Active Learning

`active_learning` is a backend-agnostic library for selecting samples by ID, attaching artifacts, and emitting results through explicit sinks.

This repository is the standalone home for the `active_learning` Python
package. The core package can be installed without `autocrane-cloud`; CRID and
Sama workflows live under explicit integration modules and require the CRID
runtime to provide the `interface` package.

## Installation

For library development:

```bash
uv pip install -e .
```

For the CRID-backed app and scripts from an `autocrane-cloud/apps/crid` shell:

```bash
uv pip install -e /Users/lukas/repos/active-learning[app,crid]
```

The app launchers expect `autocrane-cloud` next to this checkout. Set
`AUTOCRANE_CLOUD_PATH=/path/to/autocrane-cloud` if it lives elsewhere.

## Architecture

The flow is:

`source IDs -> image provider -> scorers -> selectors -> SelectionResult -> sinks`

The core rules are:

- `SampleId` is the canonical sample representation.
- Scorers derive per-sample artifacts and scores, including the cached brightness stats used for early filtering.
- Selectors choose the final subset.
- Sinks consume `SelectionResult` and handle outputs or side effects.
- Integrations adapt CRID and Sama to the core flow.

## Package Layout

- `core/`
  - shared runtime primitives: config loading, image provider, selection orchestration, and core types
- `providers/`
  - model and inference utilities: Unet loading, batch extraction, and uncertainty scoring helpers
- `scorers/`
  - score, artifact derivation, and brightness-based pre-filtering keyed by sample ID
- `selectors/`
  - final subset selection from candidates plus artifacts
- `sinks/`
  - `SelectionResult` consumers that emit outputs or side effects
- `integrations/`
  - backend-specific adapters, currently CRID and Sama, to the core flow
- `strategies/`
  - reusable selection recipes built from lower-level pieces
- `scripts/`
  - thin CLI entrypoints around the library pieces
- `tests/`
  - unit and integration coverage for the package

## Notes

- `core/config.py` owns config parsing and validation.
- `core/image_provider.py` owns image materialization and caching.
- `providers/` is separate from `core/`; it contains model/inference utilities, not image storage or CRID access.

## Strategies

The `seed.py` CLI accepts the following values for `--strategy`:

- `coreset`
  - Pure diversity selection over image features. Uses the configured feature model and any labeled seed images as the reference set.
- `uncertainty_coreset`
  - Computes uncertainty first, then balances uncertainty and diversity using coreset-style selection.
- `uncertainty_topk`
  - Pure uncertainty ranking. Selects the `n` most uncertain images without a diversity stage.
- `uncertainty_topk_coreset`
  - Two-stage uncertainty workflow: first keep the top uncertain candidates, then run coreset selection on that reduced pool.
- `alges`
  - Active Learning with Gradient Embeddings for Segmentation. Builds ALGES gradient embeddings from the configured segmentation model and selects with k-means++.
- `alges_coreset`
  - Two-stage ALGES workflow: run ALGES to form a candidate pool, then run coreset selection to diversify the final batch.

Related flags used by some strategies:

- `--provider {mc_dropout,entropy,bald}`
  - Used by the uncertainty-based strategies.
- `--aggregation {mean,topk_mean,max}` and `--topk-fraction`
  - Control how per-pixel uncertainty maps are reduced to one score per image.
- `--candidate-multiplier`
  - Used by `uncertainty_topk_coreset` to size the intermediate uncertainty shortlist.
- `--feature-model`
  - Used by `coreset`, `uncertainty_coreset`, `uncertainty_topk_coreset`, and the coreset stage of `alges_coreset`.
- `--method {image,semantic}`
  - Used by `alges` and `alges_coreset` to choose the ALGES embedding variant.

## Local Images

Use `active-learning-local` to run selection on a recursive directory of local
images without CRID or Sama:

```bash
active-learning-local --images-dir /path/to/images --strategy coreset -n 50
```

The local runner scans `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` files,
uses POSIX-style relative paths as sample IDs, and writes a mosaic plus YAML
handoff next to the configured mosaic path. `coreset` is the recommended
starter strategy because it only needs image features; uncertainty and ALGES
strategies still require a configured UNet model.

## Example

For a CRID-backed active-learning run with ALGES and Sama export:

1. `seed.py` loads the seed config, queries CRID for candidate sample IDs, and builds an `ImageProvider`.
2. `providers/` supplies the model side of the run: Unet loading, inference, and uncertainty utilities used by ALGES and uncertainty-based strategies.
3. Brightness filtering removes bad candidates, then scorers compute features, uncertainty, or ALGES embeddings.
4. A selector chooses the final `SelectionResult`.
5. `sinks/mosaic.py` can render a preview mosaic, and `sinks/yaml.py` writes the interactive seed handoff.
6. If `sama_project_id` is set, the CRID export sink submits the selected samples and the Sama sink creates the batch.

In practice, this is the path for "pick a batch of images from CRID, inspect the selection, and push it to Sama for annotation."
