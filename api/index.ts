// Vercel serverless adapter — serves the same Hono app.
import { handle } from "hono/vercel";
import app from "../src/index.js";

export const config = { runtime: "nodejs20.x" };

export default handle(app);
