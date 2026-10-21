# kaggle-colab-runner

Standalone background code-execution service for Jay's apps. Submit code,
run it on free GPUs (Kaggle) or free CPU (Wandbox), get results back.
REST API + MCP server in one service.

## Why

Train Studio, Forge 3D, and GameForge all need GPU compute but the apps
have no GPU. This service is the shared execution layer: apps submit jobs
here instead of each wiring their own Kaggle/Colab hacks.

## Backends

| Backend | What | Needs |
|---|---|---|
| `kaggle` | Free Kaggle GPU kernels (T4). Job runs async on Kaggle's infra; poll for status/output. | Server `KAGGLE_KEY` = `username:apikey` from kaggle.com -> Settings -> API |
| `cpu` | Free CPU sandbox via Wandbox. Synchronous, no key. For tests + light jobs. | Nothing |

**Colab has no public execution API.** For Colab, the service generates
ready-to-run pull-worker notebooks (`/api/worker-notebook?type=train|asset`)
pre-configured for the app's Supabase queue. Open in Colab, Run-all, leave
the tab open. Same pattern as train-studio's `ts-worker.ipynb`.

## API

- `GET /health` — service + backend status
- `POST /api/run` `{backend, code, title?, gpu?, language?}` — submit
- `GET /api/status?job_id=` — Kaggle kernel status
- `GET /api/output?job_id=` — Kaggle kernel log + parsed result
- `GET /api/quota` — Kaggle GPU quota
- `POST /api/cancel` — 501 (Kaggle's public API has no cancel; stop it on kaggle.com)
- `GET /api/jobs?backend=&state=&q=&limit=&offset=` — job ledger, newest first
- `GET /api/job?job_id=` — one ledger record
- `GET /api/languages` — CPU languages + aliases
- `GET /api/worker-notebook?type=train|asset` — Colab pull-worker .ipynb JSON
- `POST /mcp` — MCP streamable HTTP (tools: `runner_run_code`,
  `runner_job_status`, `runner_job_output`, `runner_kaggle_quota`,
  `runner_job_history`, `runner_job_detail`,
  `runner_worker_notebook`, `runner_worker_script`)

## Limits & safety

- Titles ≤ 120 chars, code ≤ 512 KB per job, job_id validated (no path junk).
- `POST /api/run` is rate-limited: 30 submissions/minute per IP (429 after).
- Kaggle API calls retry 429/5xx with exponential backoff + `Retry-After`
  (max 4 attempts); 4xx fails immediately.
- The job ledger is in-memory (resets on cold start/restart), capped at 500
  entries, 7-day TTL — an operational convenience, not an audit log.

Optional `RUNNER_API_KEY` env gates `/api/*` and `/mcp` via `x-api-key`.

## Run locally

```bash
npm install
npm test          # 20 tests, incl. real CPU execution
```

## Deploy

**Cloudflare Workers** (primary): `npx wrangler deploy` (see `wrangler.toml`).
Set secrets `KAGGLE_KEY` and `RUNNER_API_KEY`.

**Vercel**: the `api/index.ts` adapter serves the same Hono app as a
serverless function. Connect the repo in the Vercel dashboard; set the same
env vars.

## Setting the Kaggle key

1. Log in to kaggle.com -> account Settings -> API -> "Create New API Token"
   (downloads `kaggle.json` with `username` + `key`).
2. Set `KAGGLE_KEY` to `username:key` (or base64 of that) as a server secret.
3. `GET /api/quota` should return your GPU quota.

Phone verification on Kaggle is required before GPU quota is granted.

## Live URL

https://kaggle-colab-runner.jayomer1234.workers.dev
