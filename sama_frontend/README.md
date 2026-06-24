# Sama Import Frontend

Standalone Vue + Vite SPA for Sama label import, mask generation, and dataset review.

## Development

1. Start the FastAPI backend from `active_learning/api/main.py` (or use `./sama_app.sh` from the CRID shell).
2. Run `npm install`.
3. Run `npm run dev`.

The Vite dev server runs on port **5175** by default and proxies `/api/*` to `http://127.0.0.1:8080`.

## Quick start

From a CRID shell (`./run_local_uv.sh <origin> <env> --shell`):

```bash
cd /Users/lukas/repos/active-learning/sama_frontend && npm install
cd /Users/lukas/repos/active-learning && ./sama_app.sh
```
