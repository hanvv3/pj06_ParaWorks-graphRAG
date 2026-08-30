export interface RootFrontendEnvOptions {
  workspaceRoot: string;
  targetEnv?: Record<string, string | undefined>;
}

export function loadRootFrontendEnv(options: RootFrontendEnvOptions): void;
