/**
 * Minimal MCP (Model Context Protocol) streamable-HTTP handler, hand-rolled.
 * Exposes the runner's capabilities as MCP tools so agents (Error Inbox,
 * GameForge, bro-agent) can submit background GPU/CPU jobs directly.
 *
 * Supported JSON-RPC methods: initialize, notifications/initialized,
 * tools/list, tools/call, ping.
 */

import {
  acceleratorQuota,
  extractResult,
  kernelOutput,
  kernelStatus,
  parseKaggleKey,
  saveKernel,
  type KaggleCreds,
} from "./kaggle.js";
import { runCpu } from "./wandbox.js";
import { workerNotebook, workerScript, type WorkerType } from "./notebook.js";

export interface RunnerEnv {
  KAGGLE_KEY?: string;
  RUNNER_API_KEY?: string;
}

interface ToolDef {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

const TOOLS: ToolDef[] = [
  {
    name: "runner_run_code",
    description:
      "Run code in the background. backend 'kaggle' = free Kaggle GPU " +
      "(needs server KAGGLE_KEY); backend 'cpu' = free CPU sandbox (Wandbox), " +
      "no key, for tests and light jobs. Returns a job_id (Kaggle kernel ref) to poll.",
    inputSchema: {
      type: "object",
      properties: {
        backend: { type: "string", enum: ["kaggle", "cpu"], default: "cpu" },
        code: { type: "string", description: "Python code to run" },
        title: { type: "string", description: "Job title" },
        gpu: { type: "boolean", default: true, description: "Kaggle GPU (ignored by piston)" },
        language: { type: "string", default: "python" },
      },
      required: ["code"],
    },
  },
  {
    name: "runner_job_status",
    description: "Check a Kaggle job's status (queued/running/complete/error).",
    inputSchema: {
      type: "object",
      properties: { job_id: { type: "string", description: "Kernel ref from runner_run_code" } },
      required: ["job_id"],
    },
  },
  {
    name: "runner_job_output",
    description: "Fetch a Kaggle job's log output and parsed result marker.",
    inputSchema: {
      type: "object",
      properties: { job_id: { type: "string" } },
      required: ["job_id"],
    },
  },
  {
    name: "runner_kaggle_quota",
    description: "Check free Kaggle GPU accelerator quota remaining.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "runner_worker_script",
    description:
      "Get a pull-worker as plain Python (for pushing to Kaggle via " +
      "runner_run_code, or saving as .py). Same workers as " +
      "runner_worker_notebook: 'train' or 'asset'.",
    inputSchema: {
      type: "object",
      properties: {
        type: { type: "string", enum: ["train", "asset"] },
        supabase_url: { type: "string" },
        supabase_key: { type: "string" },
      },
      required: ["type"],
    },
  },
  {
    name: "runner_worker_notebook",
    description:
      "Generate a ready-to-run Colab pull-worker notebook (JSON) for a job " +
      "queue: 'train' = Train Studio QLoRA jobs, 'asset' = Forge 3D asset jobs. " +
      "Open in Colab, set the Supabase secrets, Run-all, leave the tab open.",
    inputSchema: {
      type: "object",
      properties: {
        type: { type: "string", enum: ["train", "asset"] },
        supabase_url: { type: "string" },
        supabase_key: { type: "string", description: "service_role key (worker uploads)" },
      },
      required: ["type"],
    },
  },
];

function kaggleCreds(env: RunnerEnv): KaggleCreds | null {
  return parseKaggleKey(env.KAGGLE_KEY || "");
}

function textResult(obj: unknown) {
  return { content: [{ type: "text", text: JSON.stringify(obj, null, 2) }] };
}

function toolError(msg: string) {
  return { content: [{ type: "text", text: msg }], isError: true };
}

async function callTool(
  env: RunnerEnv,
  name: string,
  args: Record<string, unknown>,
): Promise<unknown> {
  switch (name) {
    case "runner_run_code": {
      const backend = String(args.backend || "cpu");
      const code = String(args.code || "");
      if (!code.trim()) return toolError("code is required");
      if (backend === "kaggle") {
        const creds = kaggleCreds(env);
        if (!creds)
          return toolError(
            "Kaggle is not configured on the runner (no KAGGLE_KEY). " +
              "Use backend 'cpu' for CPU, or set the server's KAGGLE_KEY.",
          );
        const r = await saveKernel(creds, {
          title: String(args.title || "runner job"),
          code,
          gpu: args.gpu !== false,
        });
        if (!r.ok) return toolError(`kaggle submit failed (HTTP ${r.status}): ${r.error}`);
        return textResult({
          ok: true,
          job_id: r.ref,
          backend: "kaggle",
          hint: "poll runner_job_status(job_id)",
        });
      }
      // cpu: synchronous CPU run via Wandbox
      const r = await runCpu(code, String(args.language || "python"));
      return textResult((({ ok, ...rest }) => ({ ok, backend: "cpu", ...rest }))(r));
    }
    case "runner_job_status": {
      const creds = kaggleCreds(env);
      if (!creds) return toolError("Kaggle is not configured on the runner (no KAGGLE_KEY).");
      const r = await kernelStatus(creds, String(args.job_id || ""));
      if (!r.ok) return toolError(`status failed (HTTP ${r.status}): ${r.error}`);
      return textResult({ ok: true, job_id: args.job_id, status: r.status, detail: r });
    }
    case "runner_job_output": {
      const creds = kaggleCreds(env);
      if (!creds) return toolError("Kaggle is not configured on the runner (no KAGGLE_KEY).");
      const r = await kernelOutput(creds, String(args.job_id || ""));
      if (!r.ok) return toolError(`output failed (HTTP ${r.status}): ${r.error}`);
      return textResult({ ok: true, job_id: args.job_id, log: r.log, result: r.result });
    }
    case "runner_kaggle_quota": {
      const creds = kaggleCreds(env);
      if (!creds) return toolError("Kaggle is not configured on the runner (no KAGGLE_KEY).");
      const r = await acceleratorQuota(creds);
      if (!r.ok) return toolError(`quota failed (HTTP ${r.status}): ${r.error}`);
      return textResult({ ok: true, quota: r.quota });
    }
    case "runner_worker_script": {
      const t = String(args.type || "");
      if (t !== "train" && t !== "asset")
        return toolError("type must be 'train' or 'asset'");
      return textResult({
        ok: true,
        type: t,
        script: workerScript(t as WorkerType, {
          supabaseUrl: String(args.supabase_url || "") || undefined,
          supabaseKey: String(args.supabase_key || "") || undefined,
        }),
      });
    }
    case "runner_worker_notebook": {
      const t = String(args.type || "");
      if (t !== "train" && t !== "asset")
        return toolError("type must be 'train' or 'asset'");
      const nb = workerNotebook(t as WorkerType, {
        supabaseUrl: String(args.supabase_url || "") || undefined,
        supabaseKey: String(args.supabase_key || "") || undefined,
      });
      return textResult({
        ok: true,
        type: t,
        hint: "save as .ipynb, open in Colab, Run-all",
        notebook: nb,
      });
    }
    default:
      return toolError(`unknown tool: ${name}`);
  }
}

export async function handleMcp(env: RunnerEnv, body: unknown): Promise<Response> {
  const msg = (body || {}) as {
    jsonrpc?: string;
    id?: unknown;
    method?: string;
    params?: { name?: string; arguments?: Record<string, unknown> };
  };
  const id = msg.id ?? null;
  const ok = (result: unknown) =>
    Response.json({ jsonrpc: "2.0", id, result });
  const err = (code: number, message: string) =>
    Response.json({ jsonrpc: "2.0", id, error: { code, message } });

  if (msg.jsonrpc !== "2.0" || !msg.method) return err(-32600, "invalid request");

  switch (msg.method) {
    case "initialize":
      return ok({
        protocolVersion: "2025-03-26",
        capabilities: { tools: {} },
        serverInfo: { name: "kaggle-colab-runner", version: "1.0.0" },
      });
    case "notifications/initialized":
      return new Response(null, { status: 202 });
    case "ping":
      return ok({});
    case "tools/list":
      return ok({ tools: TOOLS });
    case "tools/call": {
      const name = msg.params?.name || "";
      const args = msg.params?.arguments || {};
      const result = await callTool(env, name, args);
      return ok(result);
    }
    default:
      return err(-32601, `method not found: ${msg.method}`);
  }
}

// Re-export for tests
export { extractResult };
