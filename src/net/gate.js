// Host-side handling of incoming connections, shared by the PeerJS and BroadcastChannel transports.
// A connection only reaches the session once its first message (a `hello`) has been authorized;
// a newly authorized connection replaces the previous one (e.g. the guest refreshed the page).

export function createHostGate(handlers) {
  let active = null;
  return {
    onData(conn, msg) {
      if (conn === active) return handlers.onMessage(msg);
      if (msg?.t !== 'hello') return;
      const verdict = handlers.authorize(msg);
      if (!verdict.ok) {
        conn.send({ t: 'reject', reason: verdict.reason });
        setTimeout(() => conn.close(), 500);
        return;
      }
      const previous = active;
      active = conn;
      previous?.close();
      handlers.onOpen();
      handlers.onMessage(msg);
    },
    onClose(conn) {
      if (conn !== active) return;
      active = null;
      handlers.onClose();
    },
    send(msg) {
      active?.send(msg);
    },
    /** Forget the current connection (used when the heartbeat says it's dead). */
    drop() {
      if (!active) return;
      const conn = active;
      active = null;
      conn.close();
      handlers.onClose();
    },
  };
}
