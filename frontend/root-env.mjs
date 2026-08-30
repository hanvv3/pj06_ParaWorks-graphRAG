import fs from "node:fs";
import path from "node:path";

import { parse } from "dotenv";

const FRONTEND_ENV_KEYS = ["NEXT_PUBLIC_API_BASE_URL", "NEXT_DIST_DIR"];

export function loadRootFrontendEnv({ workspaceRoot, targetEnv = process.env }) {
  const envPath = path.join(workspaceRoot, ".env");
  if (!fs.existsSync(envPath)) {
    return;
  }

  const rootEnv = parse(fs.readFileSync(envPath));
  for (const key of FRONTEND_ENV_KEYS) {
    if (targetEnv[key] === undefined && rootEnv[key] !== undefined) {
      targetEnv[key] = rootEnv[key];
    }
  }
}
