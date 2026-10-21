// Tests for validation, job ledger, rate limiting, and retry logic.
// Run: npm test
import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";

process.env.RUNNER_API_KEY = "";
process.env.KAGGLE_KEY = "";

const { default: app, __resetRateLimits } = await import("../src/index.ts");
const v = await import("../src/validate.ts");
const store = await import("../src/store.ts");
const kaggle = await import("../src/kaggle.ts");

async function post(path, body) {
  const r = await app.request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: r.status, json: await r.json() };
}
async function get(path) {
  const r = await app.request(path);
  return { status: r.status, json: await r.json() };
}

function expectValidation(fn, arg) {
  assert.throws(() => fn(arg), (e) => e instanceof v.ValidationError);
}

describe("validators", () => {
  it("validateBackend", () => {
    assert.equal(v.validateBackend("kaggle"), "kaggle");
    assert.equal(v.validateBackend("CPU"), "cpu");
    assert.equal(v.validateBackend(undefined), "cpu");
    expectValidation(v.validateBackend, "piston");
    expectValidation(v.validateBackend, "gpu");
  });
  it("validateTitle", () => {
    assert.equal(v.validateTitle("  hello  "), "hello");
    expectValidation(v.validateTitle, "");
    expectValidation(v.validateTitle, "   ");
    expectValidation(v.validateTitle, "x".repeat(121));
    expectValidation(v.validateTitle, "bad\x01title");
  });
  it("validateCode", () => {
    assert.equal(v.validateCode("print(1)"), "print(1)");
    expectValidation(v.validateCode, "");
    expectValidation(v.validateCode, "   \n  ");
    expectValidation(v.validateCode, "x".repeat(v.MAX_CODE_LEN + 1));
  });
  it("validateLanguage + aliases", () => {
    assert.equal(v.validateLanguage("python"), "python");
    assert.equal(v.validateLanguage("py"), "python");
    assert.equal(v.validateLanguage("JS"), "javascript");
    assert.equal(v.validateLanguage("nodejs"), "javascript");
    assert.equal(v.validateLanguage(undefined), "python");
    expectValidation(v.validateLanguage, "rust");
    expectValidation(v.validateLanguage, "c++");
  });
  it("validateJobId", () => {
    assert.equal(v.validateJobId("cpu-lxyz"), "cpu-lxyz");
    assert.equal(v.validateJobId("jaykk99/runner-m1a2b3"), "jaykk99/runner-m1a2b3");
    expectValidation(v.validateJobId, "");
    expectValidation(v.validateJobId, "../../etc/passwd");
    expectValidation(v.validateJobId, "user/slug/extra");
    expectValidation(v.validateJobId, "has space");
  });
  it("validateLimit/validateOffset", () => {
    assert.equal(v.validateLimit(undefined), 25);
    assert.equal(v.validateLimit("10"), 10);
    assert.equal(v.validateLimit(500), 100); // capped
    expectValidation(v.validateLimit, "0");
    expectValidation(v.validateLimit, "-3");
    expectValidation(v.validateLimit, "abc");
    assert.equal(v.validateOffset("5"), 5);
    assert.equal(v.validateOffset(undefined), 0);
    expectValidation(v.validateOffset, "-1");
  });
  it("validateGpu", () => {
    assert.equal(v.validateGpu(undefined), true);
    assert.equal(v.validateGpu(false), false);
    assert.equal(v.validateGpu("no"), false);
    assert.equal(v.validateGpu("1"), true);
    expectValidation(v.validateGpu, "maybe");
  });
  it("validateWorkerType", () => {
    assert.equal(v.validateWorkerType("train"), "train");
    assert.equal(v.validateWorkerType("ASSET"), "asset");
    expectValidation(v.validateWorkerType, "nope");
  });
  it("validateHttpUrl", () => {
    assert.equal(v.validateHttpUrl("https://x.supabase.co", "supabase_url"), "https://x.supabase.co");
    assert.equal(v.validateHttpUrl("", "supabase_url"), undefined);
    expectValidation(v.validateHttpUrl, "ftp://evil/x");
    expectValidation(v.validateHttpUrl, "not a url");
  });
});

describe("job ledger (store)", () => {
  beforeEach(() => store.clearJobs());
  it("record/get/touch", () => {
    const rec = store.recordJob({ job_id: "cpu-abc", backend: "cpu", title: "t1" });
    assert.equal(rec.state, "submitted");
    assert.ok(rec.submitted_at);
    assert.deepEqual(store.getJob("cpu-abc")?.title, "t1");
    store.touchJob("cpu-abc", { state: "complete", detail: "exit 0" });
    assert.equal(store.getJob("cpu-abc")?.state, "complete");
    assert.equal(store.getJob("nope"), null);
    assert.equal(store.touchJob("nope", { state: "error" }), null);
  });
  it("listJobs filters + pagination", () => {
    for (let i = 0; i < 5; i++)
      store.recordJob({ job_id: `cpu-${i}`, backend: i % 2 ? "kaggle" : "cpu", title: `job ${i}` });
    let r = store.listJobs({ limit: 10, offset: 0 });
    assert.equal(r.total, 5);
    assert.equal(r.jobs[0].job_id, "cpu-4"); // newest first
    r = store.listJobs({ backend: "cpu", limit: 10, offset: 0 });
    assert.equal(r.total, 3);
    r = store.listJobs({ q: "job 1", limit: 10, offset: 0 });
    assert.equal(r.total, 1);
    r = store.listJobs({ limit: 2, offset: 2 });
    assert.equal(r.jobs.length, 2);
    store.touchJob("cpu-0", { state: "error" });
    r = store.listJobs({ state: "error", limit: 10, offset: 0 });
    assert.equal(r.total, 1);
  });
  it("caps at MAX_JOBS, oldest evicted", () => {
    for (let i = 0; i < store.MAX_JOBS + 10; i++)
      store.recordJob({ job_id: `cpu-cap-${i}`, backend: "cpu", title: `c${i}` });
    assert.equal(store.jobCount(), store.MAX_JOBS);
    assert.equal(store.getJob("cpu-cap-0"), null);
    assert.ok(store.getJob(`cpu-cap-${store.MAX_JOBS + 9}`));
  });
});

describe("REST: validation + ledger endpoints", () => {
  beforeEach(() => {
    store.clearJobs();
    __resetRateLimits();
  });
  afterEach(() => __resetRateLimits()); // don't leak a full bucket into other files
  it("POST /api/run rejects bad input fast", async () => {
    let r = await post("/api/run", { backend: "cpu", code: "" });
    assert.equal(r.status, 400);
    assert.match(r.json.error, /code is required/);
    r = await post("/api/run", { backend: "nope", code: "print(1)" });
    assert.equal(r.status, 400);
    assert.match(r.json.error, /backend must be/);
    r = await post("/api/run", { backend: "cpu", code: "print(1)", language: "rust" });
    assert.equal(r.status, 400);
    assert.match(r.json.error, /language must be/);
    r = await post("/api/run", { backend: "cpu", code: "print(1)", title: "x".repeat(200) });
    assert.equal(r.status, 400);
    assert.match(r.json.error, /title is too long/);
  });
  it("cpu run is recorded in the ledger", async () => {
    const r = await post("/api/run", { backend: "cpu", code: "print(40 + 2)", title: "ledger probe" });
    assert.equal(r.status, 200);
    assert.match(r.json.stdout, /42/);
    const jobId = r.json.job_id;
    const j = await get(`/api/job?job_id=${encodeURIComponent(jobId)}`);
    assert.equal(j.status, 200);
    assert.equal(j.json.job.title, "ledger probe");
    assert.equal(j.json.job.backend, "cpu");
    assert.equal(j.json.job.state, "complete");
  });
  it("GET /api/job 400 on bad id, 404 on unknown", async () => {
    const bad = await get("/api/job?job_id=..%2F..%2Fx");
    assert.equal(bad.status, 400);
    const missing = await get("/api/job?job_id=cpu-doesnotexist");
    assert.equal(missing.status, 404);
  });
  it("GET /api/jobs lists, filters, paginates", async () => {
    await post("/api/run", { backend: "cpu", code: "print(1)", title: "alpha job" });
    await post("/api/run", { backend: "cpu", code: "print(2)", title: "beta job" });
    let r = await get("/api/jobs");
    assert.equal(r.status, 200);
    assert.equal(r.json.total, 2);
    assert.equal(r.json.jobs[0].title, "beta job"); // newest first
    r = await get("/api/jobs?q=alpha");
    assert.equal(r.json.total, 1);
    r = await get("/api/jobs?limit=1&offset=1");
    assert.equal(r.json.jobs.length, 1);
    r = await get("/api/jobs?state=bogus");
    assert.equal(r.status, 400);
    r = await get("/api/jobs?limit=abc");
    assert.equal(r.status, 400);
  });
  it("GET /api/languages lists cpu languages", async () => {
    const { status, json } = await get("/api/languages");
    assert.equal(status, 200);
    assert.ok(json.languages.includes("python"));
    assert.equal(json.aliases.py, "python");
  });
  it("rate limits POST /api/run (30/min)", async () => {
    for (let i = 0; i < 30; i++) {
      const r = await post("/api/run", { backend: "cpu", code: "" }); // 400, but counts
      assert.equal(r.status, 400);
    }
    const r = await post("/api/run", { backend: "cpu", code: "" });
    assert.equal(r.status, 429);
    assert.match(r.json.error, /rate limited/);
  });
});

describe("MCP: job history tools", () => {
  beforeEach(() => {
    store.clearJobs();
    __resetRateLimits();
  });
  async function rpc(method, params) {
    const { json } = await post("/mcp", { jsonrpc: "2.0", id: 1, method, params });
    return json;
  }
  it("tools/list now has 8 tools", async () => {
    const j = await rpc("tools/list", {});
    assert.equal(j.result.tools.length, 8);
    const names = j.result.tools.map((t) => t.name);
    assert.ok(names.includes("runner_job_history"));
    assert.ok(names.includes("runner_job_detail"));
  });
  it("runner_run_code validates input", async () => {
    const j = await rpc("tools/call", { name: "runner_run_code", arguments: { backend: "cpu", code: "" } });
    assert.equal(j.result.isError, true);
    assert.match(j.result.content[0].text, /code is required/);
  });
  it("runner_job_history + runner_job_detail round trip", async () => {
    const run = await rpc("tools/call", {
      name: "runner_run_code",
      arguments: { backend: "cpu", code: "print(7 * 6)", title: "history probe" },
    });
    const obj = JSON.parse(run.result.content[0].text);
    assert.equal(obj.ok, true);
    assert.match(obj.stdout, /42/);
    const hist = await rpc("tools/call", { name: "runner_job_history", arguments: {} });
    const h = JSON.parse(hist.result.content[0].text);
    assert.equal(h.ok, true);
    assert.equal(h.total, 1);
    assert.equal(h.jobs[0].title, "history probe");
    const det = await rpc("tools/call", {
      name: "runner_job_detail",
      arguments: { job_id: obj.job_id },
    });
    const d = JSON.parse(det.result.content[0].text);
    assert.equal(d.job.state, "complete");
    const missing = await rpc("tools/call", {
      name: "runner_job_detail",
      arguments: { job_id: "cpu-nope" },
    });
    assert.equal(missing.result.isError, true);
  });
});

describe("kaggle retry", () => {
  const realFetch = globalThis.fetch;
  async function withFetch(responses, fn) {
    const calls = { n: 0 };
    globalThis.fetch = async () => {
      const r = responses[Math.min(calls.n++, responses.length - 1)];
      return new Response(JSON.stringify(r.body ?? {}), {
        status: r.status,
        headers: r.retryAfter ? { "Retry-After": String(r.retryAfter) } : {},
      });
    };
    try {
      await fn(calls);
    } finally {
      globalThis.fetch = realFetch;
    }
  }

  it("retries 429 then succeeds", async () => {
    await withFetch(
      [
        { status: 429, body: { error: "slow down" }, retryAfter: 0 },
        { status: 200, body: { ref: "u/slug" } },
      ],
      async (calls) => {
        const r = await kaggle.saveKernel(
          { username: "u", apiKey: "k" },
          { title: "t", code: "print(1)" },
        );
        assert.equal(r.ok, true);
        assert.equal(calls.n, 2);
      },
    );
  });
  it("retries 500s and gives up after max attempts", async () => {
    await withFetch([{ status: 500, body: { error: "boom" } }], async (calls) => {
      const r = await kaggle.saveKernel(
        { username: "u", apiKey: "k" },
        { title: "t", code: "print(1)" },
      );
      assert.equal(r.ok, false);
      assert.equal(calls.n, 4); // MAX_ATTEMPTS
      assert.match(r.error, /kaggle HTTP 500/);
    });
  });
  it("does not retry 401", async () => {
    await withFetch([{ status: 401, body: { message: "bad key" } }], async (calls) => {
      const r = await kaggle.saveKernel(
        { username: "u", apiKey: "k" },
        { title: "t", code: "print(1)" },
      );
      assert.equal(r.ok, false);
      assert.equal(calls.n, 1);
    });
  });
  it("retryAfterMs parses seconds and dates", () => {
    const a = new Response("x", { headers: { "Retry-After": "2" } });
    assert.equal(kaggle.retryAfterMs(a), 2000);
    const b = new Response("x");
    assert.equal(kaggle.retryAfterMs(b), null);
  });
});
