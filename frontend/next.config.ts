import path from "node:path";

import type { NextConfig } from "next";

import { loadRootFrontendEnv } from "./root-env.mjs";

loadRootFrontendEnv({ workspaceRoot: path.resolve(__dirname, "..") });

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const DIST_DIR = process.env.NEXT_DIST_DIR ?? ".next";

const nextConfig: NextConfig = {
  distDir: DIST_DIR,
  outputFileTracingRoot: path.resolve(__dirname),
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_BASE}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
