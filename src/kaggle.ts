/**
 * Kaggle Kernels API client (REST, api.kaggle.com/v1).
 * Auth: Basic base64("username:apiKey") — the API token from
 * kaggle.com -> account Settings -> "Create New API Token".
 *
 * This is the same API surface train-studio's api/kaggle.js passes through:
 * SaveKernel / GetKernelSessionStatus / ListKernelSessionOutput /
 * GetAcceleratorQuotaStatistics.
 */

const BASE = "https://api.kaggle.com/v1/kernels.KernelsApiService/";

export interface KaggleCreds {
  username: string;
  apiKey: string;
}

/** Base64 decode (ASCII-safe). Works in Workers (atob) and Node (Buffer). */
function b64decode(s: string): string {
  try {
    if (typeof atob === "function") return atob(s);
  } catch { /* fall through */ }
  return Buffer.from(s, "base64").toString("utf8");
}

/** Base64 encode (ASCII-safe). Works in Workers (btoa) and Node (Buffer). */
function b64encode(s: string): string {
  if (typeof btoa === "function") return btoa(s);
  return Buffer.from(s, "utf8").toString("base64");
}

/** Parse "username:apikey" (raw or base64) into creds. */
export function parseKaggleKey(raw: string): KaggleCreds | null {
  const s = (raw || "").trim();
  if (!s) return null;
  let decoded = s;
  // If it looks like base64 "user:key", decode it.
  if (/^[A-Za-z0-9+/=]+$/.test(s) && !s.includes(":")) {
    try {
      const d = b64decode(s);
      if (d.includes(":")) decoded = d;
    } catch {
      /* not base64, use raw */
    }
  }
  const i = decoded.indexOf(":");
  if (i < 1) return null;
  const username = decoded.slice(0, i).trim();
  const apiKey = decoded.slice(i + 1).trim();
  if (!username || !apiKey) return null;
  return { username, apiKey };
}

function authHeader(c: KaggleCreds): string {
  return "Basic " + b64encode(`${c.username}:${c.apiKey}`);
}

/**
 * Retry policy for Kaggle API calls.
 * Retries 429 (rate limited) and 5xx with exponential backoff + jitter.
 * 4xx (bad request / auth) fails immediately — retrying won't help.
 */
export const RETRYABLE_STATUS = new Set([408, 409, 425, 429, 500, 502, 503, 504]);
const MAX_ATTEMPTS = 4;
const BASE_DELAY_MS = 800;

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function backoffDelay(attempt: number): number {
  // attempt is 1-based for the retry that is about to happen
  const exp = BASE_DELAY_MS * 2 ** (attempt - 1);
  const jitter = Math.random() * 400;
  return Math.min(exp + jitter, 15_000);
}

/** Parse a Retry-After header (seconds or HTTP date) into ms. */
export function retryAfterMs(res: Response): number | null {
  const v = res.headers.get("retry-after");
  if (!v) return null;
  const secs = Number(v);
  if (Number.isFinite(secs) && secs >= 0) return Math.min(secs * 1000, 60_000);
  const date = Date.parse(v);
  if (!Number.isNaN(date)) return Math.max(0, Math.min(date - Date.now(), 60_000));
  return null;
}

async function call<T>(
  creds: KaggleCreds,
  method: string,
  payload: Record<string, unknown>,
): Promise<{ ok: boolean; status: number; data: T | { error: string } }> {
  let lastErr = "";
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const outcome = await callOnce<T>(creds, method, payload);
    if (outcome.ok) return { ok: outcome.ok, status: outcome.status, data: outcome.data };
    lastErr = (outcome.data as { error?: string }).error || "";
    const retryable =
      outcome.status === 502 && /could not reach kaggle/i.test(lastErr)
        ? true // network-level failure: always retryable
        : RETRYABLE_STATUS.has(outcome.status);
    if (!retryable || attempt === MAX_ATTEMPTS)
      return { ok: outcome.ok, status: outcome.status, data: outcome.data };
    // Honor the server's Retry-After when present, else exponential backoff.
    await sleep(outcome.retryAfterMs ?? backoffDelay(attempt));
  }
  // Unreachable, but keeps TS happy.
  return { ok: false, status: 502, data: { error: lastErr } };
}

async function callOnce<T>(
  creds: KaggleCreds,
  method: string,
  payload: Record<string, unknown>,
): Promise<{ ok: boolean; status: number; data: T | { error: string }; retryAfterMs: number | null }> {
  let res: Response;
  try {
    res = await fetch(BASE + method, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        Authorization: authHeader(creds),
        "User-Agent": "kaggle-colab-runner/1.0",
      },
      body: JSON.stringify(payload || {}),
    });
  } catch (e) {
    return {
      ok: false,
      status: 502,
      data: { error: "could not reach Kaggle: " + (e as Error).message },
      retryAfterMs: null,
    };
  }
  const text = await res.text();
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    data = { error: text.slice(0, 600) || `HTTP ${res.status}` };
  }
  return { ok: res.ok, status: res.status, data: data as T, retryAfterMs: retryAfterMs(res) };
}

export interface KernelSessionStatus {
  status?: string;
  failureMessage?: string;
  [k: string]: unknown;
}

export interface KernelOutput {
  log?: string;
  [k: string]: unknown;
}

/** Wrap user code in a notebook that captures stdout/stderr and prints a
 *  RUNNER_RESULT marker with a JSON summary at the end. */
export function buildJobNotebook(title: string, code: string): string {
  const safeTitle = title.replace(/"/g, "'");
  const wrapped = [
    "import sys, json, time, traceback",
    "_t0 = time.time()",
    "_buf_out, _buf_err = [], []",
    "class _Tee:",
    "    def __init__(self, orig, buf): self.o, self.b = orig, buf",
    "    def write(self, s): self.o.write(s); self.b.append(s)",
    "    def flush(self): self.o.flush()",
    "_so, _se = sys.stdout, sys.stderr",
    "sys.stdout, sys.stderr = _Tee(_so, _buf_out), _Tee(_se, _buf_err)",
    "_exc = None",
    "try:",
    ...code.split("\n").map((l) => "    " + l),
    "except Exception as e:",
    "    _exc = traceback.format_exc()",
    "    print(_exc, file=_se)",
    "finally:",
    "    sys.stdout, sys.stderr = _so, _se",
    '    print("\\nRUNNER_RESULT " + json.dumps({',
    '        "ok": _exc is None,',
    '        "error": (_exc or "")[:4000],',
    '        "seconds": round(time.time() - _t0, 1),',
    '    }))',
  ].join("\n");

  const nb = {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
      language_info: { name: "python", version: "3.10.0" },
    },
    cells: [
      {
        cell_type: "markdown",
        metadata: {},
        source: [`# ${safeTitle}\n`, "\n", "Submitted via kaggle-colab-runner.\n"],
      },
      { cell_type: "code", metadata: {}, execution_count: null, outputs: [], source: wrapped.split("\n").map((l) => l + "\n") },
    ],
  };
  return JSON.stringify(nb);
}

/** Launch a kernel running `code`. Returns the kernel ref (username/slug). */
export async function saveKernel(
  creds: KaggleCreds,
  opts: { title: string; code: string; gpu?: boolean; internet?: boolean },
) {
  const slug =
    "runner-" +
    Date.now().toString(36) +
    "-" +
    Math.random().toString(36).slice(2, 8);
  const ref = `${creds.username}/${slug}`;
  const payload = {
    id: ref,
    slug,
    newTitle: opts.title.slice(0, 100) || "runner job",
    text: buildJobNotebook(opts.title, opts.code),
    language: "python",
    kernelType: "notebook",
    isPrivate: true,
    enableGpu: opts.gpu !== false,
    enableTpu: false,
    enableInternet: opts.internet !== false,
    datasetDataSources: [],
    competitionDataSources: [],
    kernelDataSources: [],
    categoryIds: [],
  };
  const r = await call<{ ref?: string }>(creds, "SaveKernel", payload);
  if (!r.ok) return { ok: false as const, status: r.status, error: errOf(r.data, r.status), ref: "" };
  const returnedRef =
    (r.data as { ref?: string }).ref || (r.data as { id?: string }).id || ref;
  return { ok: true as const, ref: String(returnedRef) };
}

export async function kernelStatus(creds: KaggleCreds, ref: string) {
  const { userName, kernelSlug } = splitRef(ref);
  const r = await call<KernelSessionStatus>(creds, "GetKernelSessionStatus", {
    userName,
    kernelSlug,
  });
  if (!r.ok) return { ok: false as const, status: r.status, error: errOf(r.data, r.status) };
  return { ok: true as const, ...(r.data as object) } as {
    ok: true;
    status?: string;
    [k: string]: unknown;
  };
}

export async function kernelOutput(creds: KaggleCreds, ref: string) {
  const { userName, kernelSlug } = splitRef(ref);
  const r = await call<KernelOutput>(creds, "ListKernelSessionOutput", {
    userName,
    kernelSlug,
  });
  if (!r.ok) return { ok: false as const, status: r.status, error: errOf(r.data, r.status) };
  const log = String((r.data as KernelOutput).log || "");
  return { ok: true as const, log: trimLog(log), result: extractResult(log) };
}

export async function acceleratorQuota(creds: KaggleCreds) {
  const r = await call<Record<string, unknown>>(creds, "GetAcceleratorQuotaStatistics", {});
  if (!r.ok) return { ok: false as const, status: r.status, error: errOf(r.data, r.status) };
  return { ok: true as const, quota: r.data };
}

function splitRef(ref: string): { userName: string; kernelSlug: string } {
  const i = ref.lastIndexOf("/");
  return {
    userName: i > 0 ? ref.slice(0, i) : "",
    kernelSlug: i > 0 ? ref.slice(i + 1) : ref,
  };
}

function errOf(data: unknown, status?: number): string {
  // Surface the real upstream cause: HTTP status + a compact body snippet.
  // (Kaggle error bodies never contain our API key, so this is safe to expose.)
  let detail = "";
  if (data && typeof data === "object") {
    const d = data as Record<string, unknown>;
    if (typeof d.error === "string") detail = d.error.slice(0, 300);
    else if (typeof d.message === "string") detail = d.message.slice(0, 300);
    else {
      try {
        detail = JSON.stringify(d).slice(0, 300);
      } catch {
        detail = "";
      }
      if (detail === "{}" || detail === "null" || detail === "[]") detail = "";
    }
  } else if (typeof data === "string" && data) {
    detail = data.slice(0, 300);
  }
  const prefix = status ? `kaggle HTTP ${status}` : "kaggle request failed";
  return detail ? `${prefix}: ${detail}` : prefix;
}

const MAX_LOG = 120_000;

function trimLog(log: string): string {
  if (log.length <= MAX_LOG) return log;
  const marker = log.lastIndexOf("RUNNER_RESULT");
  const tail = marker >= 0 ? log.slice(marker) : "";
  return (
    "[log trimmed — head]\n" +
    log.slice(0, 20_000) +
    "\n…[trimmed]…\n" +
    log.slice(-80_000) +
    (tail && !log.slice(-80_000).includes("RUNNER_RESULT") ? "\n" + tail : "")
  );
}

/** Pull the RUNNER_RESULT JSON marker out of a kernel log. */
export function extractResult(
  log: string,
): { ok: boolean; error: string; seconds: number } | null {
  const i = log.lastIndexOf("RUNNER_RESULT");
  if (i < 0) return null;
  const jsonPart = log.slice(i + "RUNNER_RESULT".length).trim().split("\n")[0];
  try {
    const o = JSON.parse(jsonPart);
    return {
      ok: o.ok === true,
      error: String(o.error || ""),
      seconds: Number(o.seconds || 0),
    };
  } catch {
    return null;
  }
}
