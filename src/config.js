// Deployment settings. Edit this file to use your own signaling or TURN servers.

export const CONFIG = {
  // PeerJS client library (loaded only for online games). Pinned version + integrity hash.
  peerjsUrl: 'https://cdn.jsdelivr.net/npm/peerjs@1.5.5/dist/peerjs.min.js',
  peerjsIntegrity: 'sha384-x0YgkOr/3UOZP2CRDxGW9e0Q+2Qjyr3uJrm4xU32Y7ZCNAo7Cc7bjhrZMi/dwczu',

  // Room ids live in a namespace shared by every PeerJS user, hence the prefix.
  peerIdPrefix: 'labyrinthe-duel-v1-',

  // Options passed to `new Peer(id, options)`. Empty = free public PeerJS cloud server.
  // Self-hosted example: { host: 'peer.example.com', port: 443, path: '/', secure: true }
  peerOptions: {
    debug: 1,
  },

  // Relay (TURN) servers for networks that block direct connections, e.g. school or office Wi-Fi.
  // URL returning ICE servers as JSON (an array, or { iceServers }), fetched when a game opens and
  // added to the defaults. Metered (free plan): https://<your-app>.metered.live/api/v1/turn/credentials?apiKey=<key>
  // Leave empty to use only the defaults (Google STUN + PeerJS's relay on port 3478).
  iceServersUrl: '',
};
