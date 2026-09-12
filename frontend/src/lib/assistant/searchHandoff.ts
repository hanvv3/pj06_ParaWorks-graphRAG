export type EphemeralSearchHandoff = {
  put(raw: string): void;
  consume(): string | null;
  subscribe(listener: () => void): () => void;
};

type SearchHandoffState = {
  pendingSearchInput: string | null;
  listeners: Set<() => void>;
};

const STATE_KEY = Symbol.for("paraworks.ephemeral-search-handoff");
const HANDOFF_EVENT = "paraworks:ephemeral-search-handoff-ready";

function state(): SearchHandoffState {
  const existing = Reflect.get(globalThis, STATE_KEY) as SearchHandoffState | undefined;
  if (existing) return existing;
  const created: SearchHandoffState = {
    pendingSearchInput: null,
    listeners: new Set(),
  };
  Object.defineProperty(globalThis, STATE_KEY, {
    value: created,
    configurable: true,
  });
  return created;
}

export const ephemeralSearchHandoff: EphemeralSearchHandoff = {
  put(raw) {
    const handoffState = state();
    handoffState.pendingSearchInput = raw;
    if (typeof window !== "undefined") {
      window.dispatchEvent(new Event(HANDOFF_EVENT));
    } else {
      for (const listener of handoffState.listeners) listener();
    }
  },
  consume() {
    const handoffState = state();
    const pending = handoffState.pendingSearchInput;
    handoffState.pendingSearchInput = null;
    return pending;
  },
  subscribe(listener) {
    if (typeof window !== "undefined") {
      window.addEventListener(HANDOFF_EVENT, listener);
      return () => window.removeEventListener(HANDOFF_EVENT, listener);
    }
    const handoffState = state();
    handoffState.listeners.add(listener);
    return () => handoffState.listeners.delete(listener);
  },
};
