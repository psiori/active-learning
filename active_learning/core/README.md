# Core

Core orchestration for active-learning runs.

- `config.py`: loads YAML and CLI configuration into the shared seed config model.
- `image_provider.py`: fetches and caches images behind a backend-agnostic provider API.
- `selection.py`: wires brightness filtering, scorers, providers, and selectors into runnable strategies.
- `types.py`: shared lightweight result and ID types.

Start here when you want to understand the end-to-end control flow.
