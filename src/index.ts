/**
 * kaggle-colab-runner — standalone background code-execution service.
 *
 * REST API + MCP server (streamable HTTP at /mcp).
 * Backends: kaggle (free GPU, needs KAGGLE_KEY) and cpu (free CPU via Wandbox, no key).
 * Colab: no public execution API exists — the service generates ready-to-run
 * pull-worker notebooks for Jay's Supabase job queues instead.
 *
 * Env: RUNNER_API_KEY (optional gate), KAGGLE_KEY ("username:apikey").
 */
import { Hono } from "hono";
import {
  acceleratorQuota,
  kernelOutput,
  kernelStatus,
  parseKaggleKey,
  saveKernel,
} from "./kaggle.js";
import { runCpu } from "./wandbox.js";
import { workerNotebook, workerScript, type WorkerType } from "./notebook.js";
import { handleMcp } from "./mcp.js";
import {
  validateBackend,
  validateCode,
  validateGpu,
  validateHttpUrl,
  validateJobId,
  validateLanguage,
  validateLimit,
  validateOffset,
  validateTitle,
  validateWorkerType,
  ValidationError,
} from "./validate.js";
import {
  getJob,
  jobCount,
  listJobs,
  recordJob,
  touchJob,
  type JobBackend,
  type JobState,
} from "./store.js";

type Env = {
  RUNNER_API_KEY?: string;
  KAGGLE_KEY?: string;
};

const app = new Hono<{ Bindings: Env }>();

function envOf(c: { env: Env }): Env {
  // Works on Workers (c.env), Node (process.env), and Vercel.
  // Guard process: it does not exist in the Workers runtime.
  const e = (c.env || {}) as Env;
  const penv =
    typeof process !== "undefined" && process.env ? process.env : ({} as Record<string, string | undefined>);
  return {
    RUNNER_API_KEY: e.RUNNER_API_KEY || penv.RUNNER_API_KEY,
    KAGGLE_KEY: e.KAGGLE_KEY || penv.KAGGLE_KEY,
  };
}

// API-key gate: open when RUNNER_API_KEY is unset.
app.use("/api/*", async (c, next) => {
  const key = envOf(c).RUNNER_API_KEY;
  if (key && c.req.header("x-api-key") !== key)
    return c.json({ ok: false, error: "unauthorized" }, 401);
  await next();
});
app.use("/mcp", async (c, next) => {
  const key = envOf(c).RUNNER_API_KEY;
  if (key && c.req.header("x-api-key") !== key)
    return c.json({ ok: false, error: "unauthorized" }, 401);
  await next();
});

const kaggleOn = (c: { env: Env }) => parseKaggleKey(envOf(c as never).KAGGLE_KEY || "");

// Hono types status codes as literals; our upstream HTTP codes are numbers.
function jerr(c: { json: (o: object, s?: number) => Response }, obj: object, status: number) {
  return (c.json as (o: object, s?: never) => Response)(obj, status as never);
}

/** Map a ValidationError (or any throw) to a JSON error response. */
function badRequest(c: { json: (o: object, s?: number) => Response }, e: unknown): Response {
  if (e instanceof ValidationError)
    return (c.json as (o: object, s?: never) => Response)({ ok: false, error: e.message }, e.status as never);
  throw e;
}

// ---- rate limiting ----
// Simple in-memory token bucket on POST /api/run so one client cannot flood
// Kaggle (or Wandbox) with submissions. Per-IP, 30 submits per minute.
// In-memory like the job ledger: resets on cold start / restart.

const RUN_LIMIT = 30;
const RUN_WINDOW_MS = 60_000;
const runBuckets = new Map<string, { count: number; resetAt: number }>();

function clientIp(c: { req: { header: (n: string) => string | undefined } }): string {
  return (
    c.req.header("cf-connecting-ip") ||
    c.req.header("x-forwarded-for")?.split(",")[0]?.trim() ||
    "local"
  );
}

function checkRunRateLimit(c: {
  req: { header: (n: string) => string | undefined };
}): string | null {
  const ip = clientIp(c);
  const now = Date.now();
  const b = runBuckets.get(ip);
  if (!b || now >= b.resetAt) {
    runBuckets.set(ip, { count: 1, resetAt: now + RUN_WINDOW_MS });
    return null;
  }
  if (b.count >= RUN_LIMIT) {
    const retryAfter = Math.ceil((b.resetAt - now) / 1000);
    return `rate limited: max ${RUN_LIMIT} submissions per minute (retry in ${retryAfter}s)`;
  }
  b.count++;
  return null;
}

/** Test-only: reset rate-limit buckets. */
export function __resetRateLimits(): void {
  runBuckets.clear();
}

// ---- public ----

app.get("/health", (c) => {
  const env = envOf(c);
  return c.json({
    ok: true,
    service: "kaggle-colab-runner",
    version: "1.0.0",
    backends: {
      kaggle: parseKaggleKey(env.KAGGLE_KEY || "") ? "configured" : "not configured (set KAGGLE_KEY)",
      cpu: "ready (wandbox)",
      colab: "worker-notebook generator",
    },
    jobs_tracked: jobCount(),
    limits: {
      max_code_kb: 512,
      max_title_chars: 120,
      run_rate_limit_per_min: RUN_LIMIT,
    },
    time: new Date().toISOString(),
  });
});

// CPU languages supported by the cpu backend (alias-normalized).
app.get("/api/languages", (c) =>
  c.json({
    ok: true,
    backend: "cpu",
    languages: ["python", "javascript"],
    aliases: { py: "python", python3: "python", js: "javascript", node: "javascript", nodejs: "javascript" },
  }),
);

// ---- run a job ----
// POST /api/run { backend: "kaggle"|"cpu", code, title, gpu, language }

app.post("/api/run", async (c) => {
  let body: Record<string, unknown> = {};
  try {
    body = await c.req.json();
  } catch {
    return c.json({ ok: false, error: "invalid JSON body" }, 400);
  }
  const rateMsg = checkRunRateLimit(c);
  if (rateMsg)
    return c.json({ ok: false, error: rateMsg }, 429);

  let backend: "kaggle" | "cpu";
  let title: string;
  let code: string;
  try {
    backend = validateBackend(body.backend);
    title = validateTitle(body.title);
    code = validateCode(body.code);
  } catch (e) {
    return badRequest(c, e);
  }
  const source = clientIp(c);

  if (backend === "kaggle") {
    const creds = kaggleOn(c);
    if (!creds)
      return c.json(
        {
          ok: false,
          error:
            "Kaggle is not configured on this runner (no KAGGLE_KEY). " +
            "Set the server's KAGGLE_KEY (kaggle.com -> Settings -> API) or use backend 'cpu'.",
        },
        503,
      );
    let gpu: boolean;
    try {
      gpu = validateGpu(body.gpu);
    } catch (e) {
      return badRequest(c, e);
    }
    const r = await saveKernel(creds, { title, code, gpu });
    if (!r.ok) {
      recordJob({ job_id: `kaggle-failed-${Date.now().toString(36)}`, backend, title, gpu, source, error: r.error, state: "error" });
      return jerr(c, { ok: false, error: `kaggle submit failed: ${r.error}` }, r.status || 502);
    }
    recordJob({ job_id: r.ref, backend, title, gpu, source });
    return c.json({
      ok: true,
      job_id: r.ref,
      backend: "kaggle",
      hint: "poll GET /api/status?job_id=<job_id>",
    });
  }

  // cpu: synchronous CPU run via Wandbox
  let language: "python" | "javascript";
  try {
    language = validateLanguage(body.language);
  } catch (e) {
    return badRequest(c, e);
  }
  const r = await runCpu(code, language);
  const { ok: _ok2, ...rest } = r;
  const jobId = `cpu-${Date.now().toString(36)}`;
  recordJob({
    job_id: jobId,
    backend,
    title,
    language,
    source,
    state: r.ok ? "complete" : "error",
    detail: `exit ${r.exitCode}`,
    error: r.error,
  });
  return c.json({ ok: _ok2, backend: "cpu", job_id: jobId, ...rest });
});

// ---- job ledger ----
// GET /api/jobs?backend=kaggle|cpu&state=&q=&limit=&offset=
// GET /api/job?job_id=

app.get("/api/jobs", (c) => {
  try {
    const backend = c.req.query("backend");
    const state = c.req.query("state");
    const q = c.req.query("q") || undefined;
    const limit = validateLimit(c.req.query("limit"));
    const offset = validateOffset(c.req.query("offset"));
    const b = backend ? validateBackend(backend) : undefined;
    const validStates = ["submitted", "running", "complete", "error", "unknown"];
    if (state && !validStates.includes(state))
      return c.json(
        { ok: false, error: `state must be one of: ${validStates.join(", ")}` },
        400,
      );
    const { total, jobs } = listJobs({
      backend: b as JobBackend | undefined,
      state: (state as JobState) || undefined,
      q,
      limit,
      offset,
    });
    return c.json({ ok: true, total, limit, offset, jobs });
  } catch (e) {
    return badRequest(c, e);
  }
});

app.get("/api/job", (c) => {
  try {
    const job_id = validateJobId(c.req.query("job_id") || "");
    const rec = getJob(job_id);
    if (!rec) return c.json({ ok: false, error: `unknown job_id '${job_id}'` }, 404);
    return c.json({ ok: true, job: rec });
  } catch (e) {
    return badRequest(c, e);
  }
});

// ---- job status / output (kaggle) ----

app.get("/api/status", async (c) => {
  let jobId: string;
  try {
    jobId = validateJobId(c.req.query("job_id") || "");
  } catch (e) {
    return badRequest(c, e);
  }
  const creds = kaggleOn(c);
  if (!creds) return c.json({ ok: false, error: "Kaggle is not configured (no KAGGLE_KEY)." }, 503);
  const r = await kernelStatus(creds, jobId);
  if (!r.ok) return jerr(c, { ok: false, error: r.error }, r.status || 502);
  const statusStr = String(r.status || "").toLowerCase();
  const state: JobState =
    /complete|success/.test(statusStr) ? "complete"
    : /fail|error/.test(statusStr) ? "error"
    : /run/.test(statusStr) ? "running"
    : "unknown";
  touchJob(jobId, { state, detail: String(r.status || "") });
  const { ok: _ok1, ...detail } = r;
  return c.json({ ok: true, job_id: jobId, ...detail });
});

app.get("/api/output", async (c) => {
  let jobId: string;
  try {
    jobId = validateJobId(c.req.query("job_id") || "");
  } catch (e) {
    return badRequest(c, e);
  }
  const creds = kaggleOn(c);
  if (!creds) return c.json({ ok: false, error: "Kaggle is not configured (no KAGGLE_KEY)." }, 503);
  const r = await kernelOutput(creds, jobId);
  if (!r.ok) return jerr(c, { ok: false, error: r.error }, r.status || 502);
  if (r.result) touchJob(jobId, { state: r.result.ok ? "complete" : "error", detail: "output fetched" });
  return c.json({ ok: true, job_id: jobId, log: r.log, result: r.result });
});

app.get("/api/quota", async (c) => {
  const creds = kaggleOn(c);
  if (!creds) return c.json({ ok: false, error: "Kaggle is not configured (no KAGGLE_KEY)." }, 503);
  const r = await acceleratorQuota(creds);
  if (!r.ok) return jerr(c, { ok: false, error: r.error }, r.status || 502);
  return c.json({ ok: true, quota: r.quota });
});

// Kaggle's public kernels API exposes no cancel endpoint — honest 501.
app.post("/api/cancel", async (c) => {
  return c.json(
    {
      ok: false,
      error:
        "Cancel is not supported: Kaggle's public kernels API has no cancel endpoint. " +
        "Stop the kernel from kaggle.com (your code page -> Stop).",
    },
    501,
  );
});

// ---- colab pull-worker notebook ----
// GET /api/worker-notebook?type=train|asset&supabase_url=..&supabase_key=..

app.get("/api/worker-notebook", async (c) => {
  let t: WorkerType;
  try {
    t = validateWorkerType(c.req.query("type") || "");
  } catch (e) {
    return badRequest(c, e);
  }
  const nb = workerNotebook(t, {
    supabaseUrl: validateHttpUrl(c.req.query("supabase_url"), "supabase_url"),
    supabaseKey: (c.req.query("supabase_key") || "").trim() || undefined,
  });
  return c.json({
    ok: true,
    type: t,
    hint: "save as .ipynb, open in Colab, Run-all, leave the tab open",
    notebook: nb,
  });
});

// ---- pull-worker as plain python ----
// GET /api/worker-script?type=train|asset&supabase_url=..&supabase_key=..

app.get("/api/worker-script", async (c) => {
  let t: WorkerType;
  try {
    t = validateWorkerType(c.req.query("type") || "");
  } catch (e) {
    return badRequest(c, e);
  }
  const script = workerScript(t, {
    supabaseUrl: validateHttpUrl(c.req.query("supabase_url"), "supabase_url"),
    supabaseKey: (c.req.query("supabase_key") || "").trim() || undefined,
  });
  return new Response(script, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
});

// ---- MCP (streamable HTTP) ----

app.post("/mcp", async (c) => {
  let body: unknown = {};
  try {
    body = await c.req.json();
  } catch {
    return Response.json(
      { jsonrpc: "2.0", id: null, error: { code: -32700, message: "parse error" } },
      { status: 400 },
    );
  }
  return handleMcp(envOf(c), body);
});

app.get("/mcp", (c) =>
  c.json({
    ok: true,
    mcp: "POST JSON-RPC 2.0 here (streamable HTTP)",
    methods: ["initialize", "tools/list", "tools/call", "ping"],
  }),
);

export default app;
