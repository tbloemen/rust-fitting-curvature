import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [wasm()],
  build: {
    target: "esnext",
  },
  resolve: {
    alias: {
      "@config": path.resolve(__dirname, "..", "config"),
    },
  },
});
