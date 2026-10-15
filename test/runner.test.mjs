// Tests for kaggle-colab-runner. Run: npm test
// Uses Hono's app.request() — no server needed. The piston test hits the
// real emkc.org API (free, no key).
import { describe, it } from "node:test";
import assert from "node:assert/strict";

process.env.RUNNER_API_KEY = ""; // open for tests
process.env.KAGGLE_KEY = "";

const { default: app } = await import("../src/index.ts");
const { parseKaggleKey, extractResult, buildJobNotebook } = await import("../src/kaggle.ts");
const { workerNotebook } = await import("../src/notebook.ts");

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

describe("health", () => {
  it("reports backends", async () => {
    const { status, json } = await get("/health");
    assert.equal(status, 200);
    assert.equal(json.ok, true);
    assert.equal(json.backends.cpu, "ready (wandbox)");
    assert.match(json.backends.kaggle, /not configured/);
  });
});

describe("kaggle key parsing", () => {
  it("parses user:key", () => {
    assert.deepEqual(parseKaggleKey("jaykk99:abc123"), { username: "jaykk99", apiKey: "abc123" });
  });
  it("parses base64 user:key", () => {
    const b64 = Buffer.from("jaykk99:abc123").toString("base64");
    assert.deepEqual(parseKaggleKey(b64), { username: "jaykk99", apiKey: "abc123" });
  });
  it("rejects empty", () => {
    assert.equal(parseKaggleKey(""), null);
    assert.equal(parseKaggleKey("nokey"), null);
  });
});

describe("kaggle without credentials", () => {
  it("POST /api/run kaggle -> 503", async () => {
    const { status, json } = await post("/api/run", { backend: "kaggle", code: "print(1)" });
    assert.equal(status, 503);
    assert.match(json.error, /KAGGLE_KEY/);
  });
  it("GET /api/status -> 503", async () => {
    const { status } = await get("/api/status?job_id=x/y");
    assert.equal(status, 503);
  });
  it("GET /api/quota -> 503", async () => {
    const { status } = await get("/api/quota");
    assert.equal(status, 503);
  });
});

describe("cpu backend (real execution via wandbox)", () => {
  it("runs python and returns stdout", async () => {
    const { status, json } = await post("/api/run", {
      backend: "cpu",
      code: "print(6 * 7)\nprint('hello runner')",
    });
    assert.equal(status, 200);
    assert.equal(json.ok, true);
    assert.match(json.stdout, /42/);
    assert.match(json.stdout, /hello runner/);
  });
  it("reports errors", async () => {
    const { status, json } = await post("/api/run", {
      backend: "cpu",
      code: "raise ValueError('boom')",
    });
    assert.equal(status, 200);
    assert.equal(json.ok, false);
    assert.match(json.stderr, /boom/);
  });
});

describe("job notebook builder", () => {
  it("wraps code with RUNNER_RESULT marker", () => {
    const nb = JSON.parse(buildJobNotebook("t", "print('x')"));
    assert.equal(nb.nbformat, 4);
    const src = nb.cells[1].source.join("");
    assert.match(src, /RUNNER_RESULT/);
    assert.match(src, /print\('x'\)/);
  });
  it("extractResult parses the marker", () => {
    const log = 'hello\nRUNNER_RESULT {"ok": true, "error": "", "seconds": 1.2}\n';
    assert.deepEqual(extractResult(log), { ok: true, error: "", seconds: 1.2 });
    assert.equal(extractResult("no marker here"), null);
  });
});

describe("worker notebook generator", () => {
  for (const type of ["train", "asset"]) {
    it(`${type}: valid nbformat with pull loop`, async () => {
      const nb = workerNotebook(type, { supabaseUrl: "https://x.supabase.co", supabaseKey: "k" });
      assert.equal(nb.nbformat, 4);
      const src = nb.cells.map((c) => c.source.join("")).join("\n");
      assert.match(src, /claim_job/);
      assert.match(src, /finish_job/);
      assert.match(src, /wipe\(work\)/); // disk wipe rule
      const { status, json } = await get(`/api/worker-notebook?type=${type}`);
      assert.equal(status, 200);
      assert.equal(json.notebook.nbformat, 4);
    });
  }
  it("rejects bad type", async () => {
    const { status } = await get("/api/worker-notebook?type=nope");
    assert.equal(status, 400);
  });
  it("worker-script returns plain python", async () => {
    const r = await app.request("/api/worker-script?type=asset");
    assert.equal(r.status, 200);
    const text = await r.text();
    assert.match(text, /TABLE = "forge_asset_jobs"/);
    assert.match(text, /claim_job/);
    assert.match(text, /wipe\(work\)/);
  });
});

describe("cancel honesty", () => {
  it("POST /api/cancel -> 501 with explanation", async () => {
    const { status, json } = await post("/api/cancel", { job_id: "x" });
    assert.equal(status, 501);
    assert.match(json.error, /no cancel endpoint/i);
  });
});

describe("MCP", () => {
  async function rpc(method, params) {
    const { json } = await post("/mcp", { jsonrpc: "2.0", id: 1, method, params });
    return json;
  }
  it("initialize", async () => {
    const j = await rpc("initialize", {});
    assert.equal(j.result.serverInfo.name, "kaggle-colab-runner");
  });
  it("tools/list has 6 tools", async () => {
    const j = await rpc("tools/list", {});
    assert.equal(j.result.tools.length, 6);
  });
  it("tools/call runner_worker_notebook", async () => {
    const j = await rpc("tools/call", { name: "runner_worker_notebook", arguments: { type: "train" } });
    const obj = JSON.parse(j.result.content[0].text);
    assert.equal(obj.ok, true);
    assert.equal(obj.notebook.nbformat, 4);
  });
  it("tools/call runner_run_code cpu (real)", async () => {
    const j = await rpc("tools/call", {
      name: "runner_run_code",
      arguments: { backend: "cpu", code: "print(2 + 2)" },
    });
    const obj = JSON.parse(j.result.content[0].text);
    assert.equal(obj.ok, true);
    assert.match(obj.stdout, /4/);
  });
  it("tools/call kaggle without key -> clean error", async () => {
    const j = await rpc("tools/call", {
      name: "runner_run_code",
      arguments: { backend: "kaggle", code: "print(1)" },
    });
    assert.equal(j.result.isError, true);
    assert.match(j.result.content[0].text, /KAGGLE_KEY/);
  });
});
