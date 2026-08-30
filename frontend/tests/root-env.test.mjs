import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { loadRootFrontendEnv } from "../root-env.mjs";

function withWorkspace(run) {
  const workspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "paraworks-root-env-"));
  try {
    run(workspaceRoot);
  } finally {
    fs.rmSync(workspaceRoot, { force: true, recursive: true });
  }
}

test("loads only the allowlisted frontend values from the root .env", () => {
  withWorkspace((workspaceRoot) => {
    fs.writeFileSync(
      path.join(workspaceRoot, ".env"),
      [
        "NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:9000",
        "NEXT_DIST_DIR=.next-test",
        "OPENAI_API_KEY=must-not-leave-the-server",
      ].join("\n"),
      "utf8",
    );
    const targetEnv = {};

    loadRootFrontendEnv({ workspaceRoot, targetEnv });

    assert.equal(targetEnv.NEXT_PUBLIC_API_BASE_URL, "http://127.0.0.1:9000");
    assert.equal(targetEnv.NEXT_DIST_DIR, ".next-test");
    assert.equal(targetEnv.OPENAI_API_KEY, undefined);
  });
});

test("keeps an explicit process value instead of overriding it from .env", () => {
  withWorkspace((workspaceRoot) => {
    fs.writeFileSync(
      path.join(workspaceRoot, ".env"),
      "NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:9000\n",
      "utf8",
    );
    const targetEnv = {
      NEXT_PUBLIC_API_BASE_URL: "https://api.example.test",
    };

    loadRootFrontendEnv({ workspaceRoot, targetEnv });

    assert.equal(targetEnv.NEXT_PUBLIC_API_BASE_URL, "https://api.example.test");
  });
});

test("does nothing when the root .env does not exist", () => {
  withWorkspace((workspaceRoot) => {
    const targetEnv = { EXISTING_VALUE: "preserved" };

    loadRootFrontendEnv({ workspaceRoot, targetEnv });

    assert.deepEqual(targetEnv, { EXISTING_VALUE: "preserved" });
  });
});
