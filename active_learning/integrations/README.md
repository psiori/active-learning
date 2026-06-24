# Integrations

Backend-specific adapters used by the otherwise backend-agnostic active-learning pipeline.

- `crid/`: pool queries, image downloads, dataset export, and CRID-specific metadata.
- `sama/`: batch submission helpers for Sama annotation workflows.

Keep backend API details here rather than in `core/`, `selectors/`, or `sinks/`.
