// Minimal static file server for local development: `npm start` → http://localhost:8080
// (Any static server works; this one just avoids installing anything.)

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const fixedPort = Number(process.env.PORT) || null;
let port = fixedPort ?? 8080;
const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.md': 'text/markdown; charset=utf-8',
};

const server = createServer(async (req, res) => {
  try {
    const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    let file = normalize(join(root, path));
    if (!file.startsWith(root)) throw new Error('outside root');
    if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
    res.end(body);
  } catch {
    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end('Not found');
  }
});

// Port busy (e.g. another copy already running)? Try the next one, unless PORT was set explicitly.
server.on('error', (err) => {
  if (err.code === 'EADDRINUSE' && !fixedPort && port < 8090) {
    console.log(`Port ${port} is busy, trying ${port + 1}…`);
    server.listen(++port);
  } else {
    console.error(err.code === 'EADDRINUSE' ? `Port ${port} is already in use.` : err.message);
    process.exit(1);
  }
});
server.on('listening', () => console.log(`Labyrinthe running at http://localhost:${port}`));
server.listen(port);
