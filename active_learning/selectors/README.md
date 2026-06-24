# Selectors

Selection algorithms that choose final sample IDs from scored candidates.

- `coreset.py`: diversity-focused coreset selection.
- `uncertainty.py`: uncertainty-only and hybrid uncertainty-plus-coreset selection.
- `alges.py`: ALGES selection entrypoint.
- `_helpers.py`: shared array/index alignment helpers.

Selectors operate on prepared scores or embeddings rather than raw backend APIs.
