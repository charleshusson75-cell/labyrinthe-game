// Makes sure one seat of one game is driven by a single tab of this browser at a time.
// Tabs talk over a BroadcastChannel: "who has game X?" → "I do" → "release it, I'm taking over".

const TAB_ID = Math.random().toString(36).slice(2);
const held = new Map(); // key → callback when another tab takes over
const waiting = new Map(); // key → resolver for a pending "who has" question
let bc = null;

function channel() {
  if (bc || !('BroadcastChannel' in globalThis)) return bc;
  bc = new BroadcastChannel('labyrinthe-tabs');
  bc.onmessage = ({ data }) => {
    if (!data || data.tab === TAB_ID) return;
    if (data.type === 'who-has' && held.has(data.key)) {
      bc.postMessage({ type: 'i-have', key: data.key, tab: TAB_ID, to: data.tab });
    } else if (data.type === 'i-have' && data.to === TAB_ID) {
      waiting.get(data.key)?.(true);
    } else if (data.type === 'release' && held.has(data.key)) {
      const onTakenOver = held.get(data.key);
      held.delete(data.key);
      onTakenOver();
    }
  };
  return bc;
}

/** Resolves true if another tab currently holds `key`. */
export function heldElsewhere(key, timeout = 300) {
  const c = channel();
  if (!c) return Promise.resolve(false);
  return new Promise((resolve) => {
    const done = (v) => {
      waiting.delete(key);
      clearTimeout(timer);
      resolve(v);
    };
    const timer = setTimeout(() => done(false), timeout);
    waiting.set(key, done);
    c.postMessage({ type: 'who-has', key, tab: TAB_ID });
  });
}

/** Claim `key`; `onTakenOver` runs if another tab later takes it. Returns a release function. */
export function hold(key, onTakenOver) {
  channel();
  held.set(key, onTakenOver);
  return () => held.delete(key);
}

/** Ask whichever tab holds `key` to let go. */
export function takeOver(key) {
  channel()?.postMessage({ type: 'release', key, tab: TAB_ID });
}
