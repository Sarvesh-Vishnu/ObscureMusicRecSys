/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" so the build works from any path (GitHub Pages project site or a local file server)
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { target: "es2020", sourcemap: false },
  test: { environment: "node", include: ["src/**/*.test.{ts,tsx}"] },
});
