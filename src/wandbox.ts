/**
 * CPU code-execution backend via Wandbox (https://wandbox.org).
 * Free, no API key. CPU-only — used as the always-available fallback so the
 * service can be tested end-to-end without Kaggle credentials, and for light
 * CPU jobs. GPU jobs go through the Kaggle backend.
 */

const BASE = "https://wandbox.org/api";

export interface CpuResult {
  ok: boolean;
  language: string;
  compiler: string;
  stdout: string;
  stderr: string;
  exitCode: number;
  seconds: number;
  error?: string;
}

const COMPILERS: Record<string, string> = {
  python: "cpython-3.10.15",
  javascript: "nodejs-20.17.0",
};

export async function runCpu(code: string, language = "python"): Promise<CpuResult> {
  const lang = COMPILERS[language] ? language : "python";
  const compiler = COMPILERS[lang];
  const t0 = Date.now();
  let res: Response;
  try {
    res = await fetch(BASE + "/compile.json", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ compiler, code, options: "warning" }),
    });
  } catch (e) {
    return fail(lang, compiler, "could not reach wandbox: " + (e as Error).message);
  }
  if (!res.ok) return fail(lang, compiler, `wandbox HTTP ${res.status}`);
  let data: {
    status?: string;
    program_output?: string;
    program_error?: string;
    compiler_error?: string;
    compiler_message?: string;
  };
  try {
    data = (await res.json()) as typeof data;
  } catch {
    return fail(lang, compiler, "wandbox returned invalid JSON");
  }
  const stdout = String(data.program_output || data.compiler_message || "");
  const stderr = String(data.program_error || data.compiler_error || "");
  const ok = data.status === "0" && !data.compiler_error;
  return {
    ok,
    language: lang,
    compiler,
    stdout: stdout.slice(0, 60_000),
    stderr: stderr.slice(0, 60_000),
    exitCode: ok ? 0 : 1,
    seconds: Math.round((Date.now() - t0) / 100) / 10,
  };
}

function fail(language: string, compiler: string, error: string): CpuResult {
  return { ok: false, language, compiler, stdout: "", stderr: "", exitCode: -1, seconds: 0, error };
}
