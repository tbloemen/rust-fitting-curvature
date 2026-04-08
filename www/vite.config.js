import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import path from "path";
import { createReadStream, existsSync } from "fs";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const RESULTS_DIR = path.join(__dirname, "..", "results");

const serveResults = {
  name: "serve-results",
  configureServer(server) {
    server.middlewares.use("/results", (req, res, next) => {
      const filePath = path.join(RESULTS_DIR, req.url.replace(/^\//, ""));
      if (existsSync(filePath)) {
        res.setHeader("Content-Type", "application/json");
        res.setHeader("Cache-Control", "no-cache");
        createReadStream(filePath).pipe(res);
      } else {
        next();
      }
    });
  },
};

export default defineConfig({
  plugins: [wasm(), serveResults],
  build: {
    target: "esnext",
  },
});
