/**
 * Request validation for kaggle-colab-runner.
 *
 * Every public entry point (REST + MCP) validates through these helpers so
 * bad input fails fast with a clear message instead of a confusing upstream
 * error. Limits are intentionally generous but bounded so a single request
 * cannot exhaust a Worker or blow up the Kaggle payload.
 */

export const MAX_TITLE_LEN = 120;
export const MAX_CODE_LEN = 512 * 1024; // 512 KB of source per job
export const MAX_JOB_ID_LEN = 200;
export const MAX_LIST_LIMIT = 100;
export const DEFAULT_LIST_LIMIT = 25;

export type Backend = "kaggle" | "cpu";
export type WorkerType = "train" | "asset";

export class ValidationError extends Error {
  status: number;
  constructor(message: string, status = 400) {
    super(message);
    this.name = "ValidationError";
    this.status = status;
  }
}

function fail(msg: string): never {
  throw new ValidationError(msg);
}

/** Backend must be 'kaggle' or 'cpu' (piston was removed). */
export function validateBackend(raw: unknown): Backend {
  const b = String(raw ?? "cpu").trim().toLowerCase();
  if (b === "piston")
    fail("backend 'piston' was removed (API is whitelist-only); use 'cpu'");
  if (b !== "kaggle" && b !== "cpu") fail("backend must be 'kaggle' or 'cpu'");
  return b as Backend;
}

/** Worker type must be 'train' or 'asset'. */
export function validateWorkerType(raw: unknown): WorkerType {
  const t = String(raw ?? "").trim().toLowerCase();
  if (t !== "train" && t !== "asset") fail("type must be 'train' or 'asset'");
  return t as WorkerType;
}

/** Job title: non-empty after trim, bounded length, no control chars. */
export function validateTitle(raw: unknown): string {
  const title = String(raw ?? "runner job").trim();
  if (!title) fail("title cannot be empty");
  if (title.length > MAX_TITLE_LEN)
    fail(`title is too long (max ${MAX_TITLE_LEN} characters)`);
  // eslint-disable-next-line no-control-regex
  if (/[\x00-\x1f\x7f]/.test(title)) fail("title contains control characters");
  return title;
}

/** Code payload: must be non-empty and within the size cap. */
export function validateCode(raw: unknown): string {
  const code = typeof raw === "string" ? raw : String(raw ?? "");
  if (!code.trim()) fail("code is required");
  if (code.length > MAX_CODE_LEN)
    fail(
      `code is too large (${(code.length / 1024).toFixed(0)} KB, ` +
        `max ${MAX_CODE_LEN / 1024} KB) — split the job or upload it as data`,
    );
  return code;
}

/** CPU language allowlist (matches wandbox.ts COMPILERS). */
export const CPU_LANGUAGES = ["python", "javascript"] as const;
export type CpuLanguage = (typeof CPU_LANGUAGES)[number];

const LANGUAGE_ALIASES: Record<string, CpuLanguage> = {
  python: "python",
  py: "python",
  python3: "python",
  javascript: "javascript",
  js: "javascript",
  node: "javascript",
  nodejs: "javascript",
};

export function validateLanguage(raw: unknown): CpuLanguage {
  const lang = String(raw ?? "python").trim().toLowerCase();
  const mapped = LANGUAGE_ALIASES[lang];
  if (!mapped)
    fail(`language must be one of: ${CPU_LANGUAGES.join(", ")} (got '${lang}')`);
  return mapped;
}

/**
 * Kaggle kernel ref ("user/slug") or a cpu job id ("cpu-...").
 * Rejects path traversal / injection shapes before they reach Kaggle.
 */
export function validateJobId(raw: unknown): string {
  const id = String(raw ?? "").trim();
  if (!id) fail("job_id is required");
  if (id.length > MAX_JOB_ID_LEN)
    fail(`job_id is too long (max ${MAX_JOB_ID_LEN} characters)`);
  if (/^[a-z0-9][a-z0-9_-]{0,60}$/i.test(id)) return id; // cpu-<base36>
  if (/^[a-z0-9_][a-z0-9_.-]{0,60}\/[a-z0-9][a-z0-9_-]{0,60}$/i.test(id)) return id; // user/slug
  fail(`job_id has an invalid format: '${id.slice(0, 60)}'`);
  throw new Error("unreachable");
}

/** Pagination limit: 1..MAX_LIST_LIMIT, defaults to DEFAULT_LIST_LIMIT. */
export function validateLimit(raw: unknown): number {
  if (raw === undefined || raw === null || raw === "") return DEFAULT_LIST_LIMIT;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 1)
    fail(`limit must be a positive integer (got '${raw}')`);
  return Math.min(n, MAX_LIST_LIMIT);
}

/** Pagination offset: >= 0. */
export function validateOffset(raw: unknown): number {
  if (raw === undefined || raw === null || raw === "") return 0;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0)
    fail(`offset must be a non-negative integer (got '${raw}')`);
  return n;
}

/** GPU flag: boolean-ish, defaults true for kaggle. */
export function validateGpu(raw: unknown): boolean {
  if (raw === undefined || raw === null) return true;
  if (typeof raw === "boolean") return raw;
  const s = String(raw).trim().toLowerCase();
  if (["true", "1", "yes", "y"].includes(s)) return true;
  if (["false", "0", "no", "n"].includes(s)) return false;
  fail(`gpu must be a boolean (got '${raw}')`);
  throw new Error("unreachable");
}

/** Optional URL-ish value (supabase_url etc.): must be http(s) when present. */
export function validateHttpUrl(raw: unknown, field: string): string | undefined {
  const s = String(raw ?? "").trim();
  if (!s) return undefined;
  if (!/^https?:\/\//i.test(s)) fail(`${field} must be an http(s) URL`);
  if (s.length > 500) fail(`${field} is too long`);
  return s;
}
