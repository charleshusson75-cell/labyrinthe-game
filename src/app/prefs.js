// Persistent per-browser preferences and saved games (localStorage), failing soft when storage is
// unavailable (private mode, blocked cookies…).

const PREFIX = 'labyrinthe:';
const GAME_PREFIX = `${PREFIX}game:v1:`;

function read(key) {
  try {
    const raw = localStorage.getItem(key);
    return raw == null ? undefined : JSON.parse(raw);
  } catch {
    return undefined;
  }
}

function write(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}

export const prefs = {
  get: (name, fallback) => read(PREFIX + name) ?? fallback,
  set: (name, value) => write(PREFIX + name, value),
};

/** Storage adapter for one seat of one game, as expected by createSession. */
export function gameStorage(kind, code, seat) {
  const key = `${GAME_PREFIX}${kind}:${code}:${seat}`;
  return {
    exists: () => read(key) !== undefined,
    load: () => read(key) ?? null,
    save: (state) => write(key, { ...state, savedAt: Date.now() }),
    remove: () => {
      try {
        localStorage.removeItem(key);
      } catch {
        /* ignore */
      }
    },
  };
}

/** Saved games of this browser, most recent first: [{ kind, code, seat, state }]. */
export function savedGames() {
  const out = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key?.startsWith(GAME_PREFIX)) continue;
      const [kind, code, seat] = key.slice(GAME_PREFIX.length).split(':');
      if (kind === 'bot' && seat !== 'host') continue; // the bot's half of a bot game
      const state = read(key);
      if (state) out.push({ kind, code, seat, state });
    }
  } catch {
    /* ignore */
  }
  return out.sort((a, b) => (b.state.savedAt ?? 0) - (a.state.savedAt ?? 0));
}

export function forgetGame(kind, code) {
  for (const seat of ['host', 'guest']) gameStorage(kind, code, seat).remove();
}
