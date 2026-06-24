# Active Learning Frontend

Client-rendered Vue + Vite SPA for the active-learning query preview UI.

Sama label import and mask generation live in the separate `../sama_frontend` app.

## Development

1. Start the FastAPI backend from `active_learning/api/main.py`.
2. Run `npm install`.
3. Run `npm run dev`.

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000`.
