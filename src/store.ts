/**
 * In-memory job ledger for kaggle-colab-runner.
 *
 * Records every submitted job (kaggle + cpu) with its title, backend, and
 * lifecycle timestamps so agents can list and audit recent work via
 * GET /api/jobs and the `runner_job_history` MCP tool.
 *
 * Note: this is in-memory. On Cloudflare Workers each isolate keeps its own
 * ledger and it resets on cold start — it is an operational convenience, not
 * a durable audit log. On the Node/Vercel adapter it lives as long as the
 * process. Entries expire after JOB_TTL_MS and the ledger is capped at
 * MAX_JOBS (oldest-first eviction).
 */

export type JobBackend = "kaggle" | "cpu";
export type JobState = "submitted" | "running" | "complete" | "error" | "unknown";

export interface JobRecord {
  job_id: string;
  backend: JobBackend;
  title: string;
  language?: string;
  gpu?: boolean;
  submitted_at: string; // ISO
  updated_at: string; // ISO
  state: JobState;
  /** Last known upstream detail (kaggle status string, cpu ok flag). */
  detail?: string;
  /** Client IP or API-key label that submitted it (redacted to a hash). */
  source?: string;
  error?: string;
  /** Monotonic sequence — disambiguates same-millisecond submissions. */
  seq: number;
}

export const MAX_JOBS = 500;
export const JOB_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

// Single shared ledger for the process/isolate.
const jobs = new Map<string, JobRecord>();
let seqCounter = 0;

function nowIso(): string {
  return new Date().toISOString();
}

function expired(r: JobRecord): boolean {
  return Date.now() - Date.parse(r.submitted_at) > JOB_TTL_MS;
}

function prune(): void {
  if (jobs.size <= MAX_JOBS) {
    // Still sweep expired entries on the way in.
    for (const [id, r] of jobs) if (expired(r)) jobs.delete(id);
    return;
  }
  const ordered = [...jobs.values()].sort((a, b) =>
    a.submitted_at < b.submitted_at ? -1 : 1,
  );
  for (const r of ordered.slice(0, jobs.size - MAX_JOBS)) jobs.delete(r.job_id);
  for (const [id, r] of jobs) if (expired(r)) jobs.delete(id);
}

export function recordJob(rec: Omit<JobRecord, "submitted_at" | "updated_at" | "state" | "seq"> & { state?: JobState }): JobRecord {
  const full: JobRecord = {
    ...rec,
    state: rec.state ?? "submitted",
    submitted_at: nowIso(),
    updated_at: nowIso(),
    seq: ++seqCounter,
  };
  jobs.set(full.job_id, full);
  prune();
  return full;
}

export function getJob(job_id: string): JobRecord | null {
  const r = jobs.get(job_id) ?? null;
  if (r && expired(r)) {
    jobs.delete(job_id);
    return null;
  }
  return r;
}

/** Update lifecycle state (called from /api/status polling). */
export function touchJob(
  job_id: string,
  patch: Partial<Pick<JobRecord, "state" | "detail" | "error">>,
): JobRecord | null {
  const r = jobs.get(job_id);
  if (!r) return null;
  Object.assign(r, patch, { updated_at: nowIso() });
  return r;
}

export interface ListJobsOpts {
  backend?: JobBackend;
  state?: JobState;
  q?: string;
  limit: number;
  offset: number;
}

export function listJobs(opts: ListJobsOpts): { total: number; jobs: JobRecord[] } {
  const q = (opts.q || "").toLowerCase();
  const all = [...jobs.values()]
    .filter((r) => !expired(r))
    .filter((r) => !opts.backend || r.backend === opts.backend)
    .filter((r) => !opts.state || r.state === opts.state)
    .filter(
      (r) =>
        !q ||
        r.title.toLowerCase().includes(q) ||
        r.job_id.toLowerCase().includes(q),
    )
    .sort((a, b) => b.seq - a.seq); // newest first
  return {
    total: all.length,
    jobs: all.slice(opts.offset, opts.offset + opts.limit),
  };
}

export function jobCount(): number {
  let n = 0;
  for (const r of jobs.values()) if (!expired(r)) n++;
  return n;
}

/** Test-only: reset the ledger. */
export function clearJobs(): void {
  jobs.clear();
}
