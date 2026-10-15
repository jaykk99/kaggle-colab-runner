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

type Env = {
  RUNNER_API_KEY?: string;
  KAGGLE_KEY?: string;
};

const app = new Hono<{ Bindings: Env }>();

function envOf(c: { env: Env }): Env {
  // Works on Workers (c.env), Node (process.env), and Vercel.
  const e = (c.env || {}) as Env;
  return {
    RUNNER_API_KEY: e.RUNNER_API_KEY || process.env.RUNNER_API_KEY,
    KAGGLE_KEY: e.KAGGLE_KEY || process.env.KAGGLE_KEY,
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
    time: new Date().toISOString(),
  });
});

// ---- run a job ----
// POST /api/run { backend: "kaggle"|"cpu", code, title, gpu, language }

app.post("/api/run", async (c) => {
  let body: Record<string, unknown> = {};
  try {
    body = await c.req.json();
  } catch {
    return c.json({ ok: false, error: "invalid JSON body" }, 400);
  }
  const backend = String(body.backend || "cpu");
  if (backend === "piston") return c.json({ ok: false, error: "backend 'piston' was removed (API is whitelist-only); use 'cpu'" }, 400);
  const code = String(body.code || "");
  if (!code.trim()) return c.json({ ok: false, error: "code is required" }, 400);

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
    const r = await saveKernel(creds, {
      title: String(body.title || "runner job"),
      code,
      gpu: body.gpu !== false,
    });
    if (!r.ok)
      return jerr(c, { ok: false, error: `kaggle submit failed: ${r.error}` }, r.status || 502);
    return c.json({
      ok: true,
      job_id: r.ref,
      backend: "kaggle",
      hint: "poll GET /api/status?job_id=<job_id>",
    });
  }

  if (backend === "cpu") {
    const r = await runCpu(code, String(body.language || "python"));
    const { ok: _ok2, ...rest } = r;
    return c.json({ ok: _ok2, backend: "cpu", job_id: `cpu-${Date.now().toString(36)}`, ...rest });
  }

  return c.json({ ok: false, error: "backend must be 'kaggle' or 'cpu'" }, 400);
});

// ---- job status / output (kaggle) ----

app.get("/api/status", async (c) => {
  const jobId = c.req.query("job_id") || "";
  if (!jobId) return c.json({ ok: false, error: "job_id is required" }, 400);
  const creds = kaggleOn(c);
  if (!creds) return c.json({ ok: false, error: "Kaggle is not configured (no KAGGLE_KEY)." }, 503);
  const r = await kernelStatus(creds, jobId);
  if (!r.ok) return jerr(c, { ok: false, error: r.error }, r.status || 502);
  const { ok: _ok1, ...detail } = r;
  return c.json({ ok: true, job_id: jobId, ...detail });
});

app.get("/api/output", async (c) => {
  const jobId = c.req.query("job_id") || "";
  if (!jobId) return c.json({ ok: false, error: "job_id is required" }, 400);
  const creds = kaggleOn(c);
  if (!creds) return c.json({ ok: false, error: "Kaggle is not configured (no KAGGLE_KEY)." }, 503);
  const r = await kernelOutput(creds, jobId);
  if (!r.ok) return jerr(c, { ok: false, error: r.error }, r.status || 502);
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
  const t = c.req.query("type") || "";
  if (t !== "train" && t !== "asset")
    return c.json({ ok: false, error: "type must be 'train' or 'asset'" }, 400);
  const nb = workerNotebook(t as WorkerType, {
    supabaseUrl: c.req.query("supabase_url") || undefined,
    supabaseKey: c.req.query("supabase_key") || undefined,
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
  const t = c.req.query("type") || "";
  if (t !== "train" && t !== "asset")
    return c.json({ ok: false, error: "type must be 'train' or 'asset'" }, 400);
  const script = workerScript(t as WorkerType, {
    supabaseUrl: c.req.query("supabase_url") || undefined,
    supabaseKey: c.req.query("supabase_key") || undefined,
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
