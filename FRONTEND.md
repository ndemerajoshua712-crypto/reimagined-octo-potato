# DeepTrace frontend

React + TypeScript + Vite application under `frontend/`. It connects to the FastAPI service, uploads evidence, tracks jobs with Server-Sent Events and polling fallback, lists prior cases, and displays completed findings, sampled frames, audio windows, scene segments, and downloadable reports.

## Start locally

From the repository root, make sure the backend and worker are running, then:

```powershell
Set-Location frontend
if (!(Test-Path .env)) { Copy-Item .env.example .env }
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Set `VITE_API_BASE_URL` in `frontend/.env` if the API is not at `http://127.0.0.1:8000`.

By default, the Vite development server proxies `/api` and `/ready` to the local FastAPI service on port `8000`, which also allows a forwarded frontend tunnel to reach that API through the same origin. For separately hosted services, set `VITE_API_BASE_URL` to the API origin and allow the frontend origin in the backend's `DEEPTRACE_CORS_ORIGINS` setting.

The app accepts MP4, MOV, AVI, MKV, and WebM videos. It reads the server's upload limit and rejects oversized files before sending them. The frame interval controls requested sampling density; long recordings may be sampled at a wider effective interval to keep model work bounded, which is shown in the result and coverage notes. Videos without usable audio or faces still return whatever other evidence could be analyzed, with missing signals labeled unavailable. Job updates arrive over Server-Sent Events, with status polling as a fallback. Progress is stage-based and can pause while a stage analyzes its frames.

Create a production bundle with `npm run build`; `npm run preview` serves that bundle locally.

If port `5173` is already in use, check whether the frontend is already running before starting another development server. See [README.md](README.md) for the complete local workflow and troubleshooting notes.
