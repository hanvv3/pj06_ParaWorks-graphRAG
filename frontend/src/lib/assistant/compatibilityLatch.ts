export type AssistantClientUpgradeLatch = {
  get(): boolean;
  activate(): void;
  subscribe(listener: () => void): () => void;
};

type BrowserLatchState = {
  active: boolean;
  listeners: Set<() => void>;
};

const STATE_KEY = Symbol.for("paraworks.assistant-client-upgrade-latch");

function browserState(): BrowserLatchState | undefined {
  if (typeof window === "undefined") return undefined;
  const existing = Reflect.get(window, STATE_KEY) as BrowserLatchState | undefined;
  if (existing) return existing;
  const created: BrowserLatchState = {
    active: false,
    listeners: new Set(),
  };
  Object.defineProperty(window, STATE_KEY, {
    value: created,
    configurable: true,
  });
  return created;
}

export const assistantClientUpgradeLatch: AssistantClientUpgradeLatch = {
  get() {
    return browserState()?.active ?? false;
  },
  activate() {
    const state = browserState();
    if (!state || state.active) return;
    state.active = true;
    for (const listener of [...state.listeners]) listener();
  },
  subscribe(listener) {
    const state = browserState();
    if (!state) return () => undefined;
    state.listeners.add(listener);
    if (state.active) listener();
    return () => state.listeners.delete(listener);
  },
};
