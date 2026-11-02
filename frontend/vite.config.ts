import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/datasets": "http://127.0.0.1:8000",
      "/mechanisms": "http://127.0.0.1:8000",
      "/runs": "http://127.0.0.1:8000",
      "/pareto": "http://127.0.0.1:8000",
      "/compliance": "http://127.0.0.1:8000",
      "/perturb": "http://127.0.0.1:8000",
      "/recommend": "http://127.0.0.1:8000",
    },
  },
});
