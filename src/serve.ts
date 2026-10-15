// Local dev server (node). Not used on Workers/Vercel.
import app from "./index.js";

const port = Number(process.env.PORT || 8787);
const isBun = typeof (globalThis as unknown as { Bun?: unknown }).Bun !== "undefined";

if (isBun) {
  const BunRt = (globalThis as unknown as { Bun: { serve: (o: object) => void } }).Bun;
  BunRt.serve({ port, fetch: app.fetch });
  console.log(`listening on :${port} (bun)`);
} else {
  const http = await import("node:http");
  http
    .createServer((req, res) => {
      const url = `http://${req.headers.host}${req.url}`;
      const chunks: Buffer[] = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", async () => {
        const body = Buffer.concat(chunks);
        const request = new Request(url, {
          method: req.method,
          headers: req.headers as HeadersInit,
          body: ["GET", "HEAD"].includes(req.method || "") ? undefined : body,
        });
        const response = await app.fetch(request);
        res.writeHead(response.status, Object.fromEntries(response.headers.entries()));
        const buf = Buffer.from(await response.arrayBuffer());
        res.end(buf);
      });
    })
    .listen(port, () => console.log(`listening on :${port} (node)`));
}
