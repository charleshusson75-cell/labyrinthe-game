// Same-browser transport over BroadcastChannel: lets you play (or debug) a game between two tabs
// without any network. Enabled with `?net=local` in the URL.

import { createHostGate } from './gate.js';

export function createLocalTransport({ role, roomCode }) {
  const id = Math.random().toString(36).slice(2, 10);
  let bc = null;
  let handlers = null;
  let knockTimer = null;
  let hostId = null; // guest: the host tab we're talking to
  let gate = null;
  const conns = new Map(); // host: guest tab id → connection wrapper

  const post = (data) => bc?.postMessage({ ...data, from: id });

  function hostMessage(data) {
    if (data.kind === 'knock') return post({ kind: 'welcome', to: data.from });
    let conn = conns.get(data.from);
    if (data.kind === 'msg') {
      if (!conn) {
        const to = data.from;
        conn = {
          send: (payload) => post({ kind: 'msg', to, payload }),
          close: () => post({ kind: 'close', to }),
        };
        conns.set(to, conn);
      }
      gate.onData(conn, data.payload);
    } else if (data.kind === 'bye' && conn) {
      conns.delete(data.from);
      gate.onClose(conn);
    }
  }

  function startKnocking() {
    clearInterval(knockTimer);
    post({ kind: 'knock' });
    knockTimer = setInterval(() => post({ kind: 'knock' }), 1000);
  }

  function guestMessage(data) {
    if (data.kind === 'host-up' || (data.kind === 'bye' && data.from === hostId)) {
      if (hostId) {
        hostId = null;
        handlers.onClose();
      }
      return startKnocking();
    }
    if (data.to !== id) return;
    if (data.kind === 'welcome' && !hostId) {
      hostId = data.from;
      clearInterval(knockTimer);
      handlers.onStatus('ready');
      handlers.onOpen();
    } else if (data.kind === 'msg' && data.from === hostId) {
      handlers.onMessage(data.payload);
    } else if (data.kind === 'close' && data.from === hostId) {
      hostId = null;
      handlers.onClose();
      startKnocking();
    }
  }

  const bye = () => post({ kind: 'bye' });

  return {
    start(h) {
      handlers = h;
      bc = new BroadcastChannel(`labyrinthe-local-${roomCode}`);
      globalThis.addEventListener?.('pagehide', bye);
      if (role === 'host') {
        gate = createHostGate(h);
        bc.onmessage = ({ data }) => (!data.to || data.to === id) && hostMessage(data);
        h.onStatus('ready');
        post({ kind: 'host-up' });
      } else {
        bc.onmessage = ({ data }) => guestMessage(data);
        h.onStatus('ready', { code: 'host-offline' });
        startKnocking();
      }
    },
    send(msg) {
      if (role === 'host') gate?.send(msg);
      else if (hostId) post({ kind: 'msg', to: hostId, payload: msg });
    },
    reset() {
      if (role === 'host') gate?.drop();
      else if (hostId) {
        hostId = null;
        handlers.onClose();
        startKnocking();
      }
    },
    destroy() {
      bye();
      clearInterval(knockTimer);
      globalThis.removeEventListener?.('pagehide', bye);
      bc?.close();
      bc = null;
    },
  };
}
