// In-memory transport pair: used for games against the computer and by the tests.
// Messages are JSON round-tripped and delivered asynchronously, like a real network.

export function createPipe({ delay = 0 } = {}) {
  let linked = true;
  let inFlight = 0;
  let dropCount = 0;
  const ends = [0, 1].map((i) => ({ i, handlers: null, open: false }));

  function open(end) {
    if (end.open || !end.handlers) return;
    end.open = true;
    end.handlers.onOpen();
  }

  function close(end) {
    if (!end.open) return;
    end.open = false;
    end.handlers?.onClose();
  }

  function deliver(to, msg) {
    if (!linked || !to.handlers) return;
    // The side with `authorize` acts like a host: a connection only counts once its hello is accepted.
    if (to.handlers.authorize && !to.open) {
      if (msg?.t !== 'hello') return;
      const verdict = to.handlers.authorize(msg);
      if (!verdict.ok) return transmit(to, { t: 'reject', reason: verdict.reason });
      open(to);
    }
    if (to.open) to.handlers.onMessage(msg);
  }

  function transmit(from, msg) {
    if (dropCount > 0) {
      dropCount--;
      return;
    }
    const to = ends[1 - from.i];
    const data = JSON.parse(JSON.stringify(msg));
    inFlight++;
    setTimeout(() => {
      inFlight--;
      deliver(to, data);
    }, delay);
  }

  function openDialers() {
    if (!linked) return;
    for (const end of ends) {
      if (end.handlers && !end.handlers.authorize) setTimeout(() => linked && open(end), 0);
    }
  }

  const transportFor = (end) => ({
    start(handlers) {
      end.handlers = handlers;
      handlers.onStatus?.('ready');
      openDialers();
    },
    send(msg) {
      if (linked && end.handlers) transmit(end, msg);
    },
    reset() {
      pipe.disconnect();
      pipe.connect();
    },
    destroy() {
      end.open = false;
      end.handlers = null;
      close(ends[1 - end.i]);
    },
  });

  const pipe = {
    host: transportFor(ends[0]),
    guest: transportFor(ends[1]),
    connect() {
      linked = true;
      openDialers();
    },
    disconnect() {
      linked = false;
      ends.forEach(close);
    },
    /** Silently lose the next `n` messages (either direction). */
    drop(n = 1) {
      dropCount += n;
    },
    /** Resolves once no message is in flight. */
    async settled() {
      for (let i = 0; i < 5000; i++) {
        await new Promise((r) => setTimeout(r, delay));
        if (inFlight === 0) {
          await new Promise((r) => setTimeout(r, 0));
          if (inFlight === 0) return;
        }
      }
      throw new Error('pipe never settled');
    },
  };
  return pipe;
}
