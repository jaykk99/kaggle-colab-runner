// Mocked Kaggle end-to-end: proves submit -> status -> output -> result
// parsing works, without needing real Kaggle credentials.
const realFetch = globalThis.fetch;
const calls: string[] = [];
globalThis.fetch = (async (url: string, init: never) => {
  calls.push(String(url));
  const u = String(url);
  const body = JSON.parse(String((init as { body: string }).body || "{}"));
  const json = (o: unknown) =>
    new Response(JSON.stringify(o), { status: 200, headers: { "Content-Type": "application/json" } });
  if (u.endsWith("SaveKernel")) {
    if (!body.text || !body.text.includes("RUNNER_RESULT"))
      return new Response(JSON.stringify({ error: "bad notebook" }), { status: 400 });
    if (body.enableGpu !== true) return new Response(JSON.stringify({ error: "gpu off" }), { status: 400 });
    return json({ ref: "testuser/runner-abc123", id: "testuser/runner-abc123" });
  }
  if (u.endsWith("GetKernelSessionStatus")) {
    if (body.userName !== "testuser" || body.kernelSlug !== "runner-abc123")
      return new Response(JSON.stringify({ error: "not found" }), { status: 404 });
    return json({ status: "COMPLETE" });
  }
  if (u.endsWith("ListKernelSessionOutput")) {
    return json({ log: 'hello\nRUNNER_RESULT {"ok": true, "error": "", "seconds": 4.2}\n' });
  }
  if (u.endsWith("GetAcceleratorQuotaStatistics")) return json({ quotaRemaining: 30 });
  return realFetch(url, init);
}) as typeof fetch;

const kaggle = await import("../src/kaggle.ts");
const creds = { username: "testuser", apiKey: "fake" } as never;

// 1. submit
const s = await kaggle.saveKernel(creds, { title: "mock job", code: "print('hi')", gpu: true });
console.log("submit:", s.ok, (s as { ref: string }).ref);
if (!s.ok) throw new Error("submit failed");

// 2. status
const st = await kaggle.kernelStatus(creds, (s as { ref: string }).ref);
console.log("status:", st.ok, (st as { status: string }).status);
if (!st.ok || (st as { status: string }).status !== "COMPLETE") throw new Error("status failed");

// 3. output + result parsing
const out = await kaggle.kernelOutput(creds, (s as { ref: string }).ref);
console.log("output ok:", out.ok, "result:", JSON.stringify((out as { result: unknown }).result));
if (!out.ok || !(out as { result: { ok: boolean } }).result?.ok) throw new Error("output failed");

// 4. quota
const q = await kaggle.acceleratorQuota(creds);
console.log("quota:", q.ok);

console.log("endpoints hit:", calls.map((c) => c.split("/").pop()).join(", "));
console.log("MOCKED KAGGLE E2E: PASS");
