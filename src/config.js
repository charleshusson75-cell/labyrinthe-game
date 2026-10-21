// Deployment settings. Edit this file to use your own signaling or TURN servers.

export const CONFIG = {
  // PeerJS client library (loaded only for online games). Pinned version + integrity hash.
  peerjsUrl: 'https://cdn.jsdelivr.net/npm/peerjs@1.5.5/dist/peerjs.min.js',
  peerjsIntegrity: 'sha384-x0YgkOr/3UOZP2CRDxGW9e0Q+2Qjyr3uJrm4xU32Y7ZCNAo7Cc7bjhrZMi/dwczu',

  // Room ids live in a namespace shared by every PeerJS user, hence the prefix.
  peerIdPrefix: 'labyrinthe-duel-v1-',

  // Options passed to `new Peer(id, options)`. Empty = free public PeerJS cloud server.
  // Self-hosted example: { host: 'peer.example.com', port: 443, path: '/', secure: true }
  // To help players behind strict NATs, add a TURN server:
  //   config: { iceServers: [{ urls: 'stun:stun.l.google.com:19302' },
  //                          { urls: 'turn:turn.example.com:3478', username: 'u', credential: 'p' }] }
  peerOptions: {
    debug: 1,
  },
};
