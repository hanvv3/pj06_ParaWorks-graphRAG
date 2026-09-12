export type EphemeralSearchHandoff = {
  put(raw: string): void;
  consume(): string | null;
};

let pendingSearchInput: string | null = null;

export const ephemeralSearchHandoff: EphemeralSearchHandoff = {
  put(raw) {
    pendingSearchInput = raw;
  },
  consume() {
    const pending = pendingSearchInput;
    pendingSearchInput = null;
    return pending;
  },
};
