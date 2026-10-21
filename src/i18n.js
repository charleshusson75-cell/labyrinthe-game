// UI strings in English and French.
// `t()` returns HTML-safe text (parameters are escaped; dictionary strings may hold simple markup).
// `tt()` returns plain text, for toasts, the page title and attributes.

import { prefs } from './app/prefs.js';
import { esc } from './ui/dom.js';

const plural = (n, one, many) => (n === 1 ? one : many);

const en = {
  appName: 'Labyrinthe',
  tagline: 'Draw a secret maze. Find your friend’s exit before they find yours.',
  you: 'You',
  opponent: 'Opponent',
  computer: 'Computer',
  'level.easy': 'easy',
  'level.normal': 'normal',
  'level.hard': 'hard',
  'nav.menu': 'Menu',
  'nav.sound': 'Sound on/off',
  'nav.lang': 'Français',

  'home.name': 'Your name',
  'home.namePlaceholder': 'e.g. Alex',
  'home.friend.title': 'Play a friend online',
  'home.friend.desc': 'Create a game, send the link, each of you plays on your own screen.',
  'home.size': 'Grid',
  'home.walls': 'Walls each',
  'home.wallsHint': '1 – {max}',
  'home.create': 'Create game',
  'home.bot.title': 'Play the computer',
  'home.bot.desc': 'Warm up against a bot.',
  'home.level': 'Difficulty',
  'home.start': 'Start',
  'home.join.title': 'Join a game',
  'home.join.desc': 'Got a link or a code from a friend?',
  'home.join.placeholder': 'Code or link',
  'home.join.button': 'Join',
  'home.join.invalid': 'That doesn’t look like a game code.',
  'home.saved.title': 'Your games',
  'home.saved.resume': 'Resume',
  'home.saved.forget': 'Forget',
  'home.saved.vs': 'vs {name}',
  'home.saved.waiting': 'waiting for an opponent',
  'home.saved.round': 'round {n}',
  'home.saved.host': 'you host',
  'rules.title': 'How to play',
  rules: [
    'Each player secretly draws a maze on their own grid: an exact number of <b>walls</b> (a wall is one side of a cell), an <b>entrance E</b> and an <b>exit S</b>.',
    'When both are ready, you only learn where your opponent’s E and S are.',
    'On your turn, try to move from a cell you have reached (you start on E) to a neighbouring cell.',
    '<b>Open?</b> You move and play again. <b>Wall?</b> It gets drawn on your map and the turn passes.',
    'You can continue from any cell you have already reached: dead end? Jump back to a fork.',
    'First to reach the other’s S wins. The loser starts the next round.',
    'At the end both mazes are revealed and checked against what each player locked in, so cheating gets caught.',
  ],

  'join.title': 'You’re invited!',
  'join.desc': 'Someone challenged you to a Labyrinthe duel. Pick a name and jump in.',
  'join.go': 'Join the game',

  'lock.title': 'Already open',
  'lock.desc': 'This game is open in another tab of this browser.',
  'lock.here': 'Play here instead',
  'lock.away': 'This game is now open in another tab.',

  'game.room': 'Room',
  'game.copyLink': 'Copy link',
  'game.copied': 'Link copied!',
  'game.share': 'Share…',
  'invite.title': 'Invite your opponent:',
  'invite.desc': 'send them this link.',
  'invite.meanwhile': 'Meanwhile, start drawing your maze. It stays secret.',
  'invite.shareText': 'Come and play Labyrinthe with me!',

  'status.connecting': 'Connecting to the game…',
  'status.slow': 'Still trying to connect…',
  'status.hostOffline': 'Waiting for {name} to come online…',
  'status.theHost': 'the host',
  'status.network': 'Lost the connection to the server, retrying…',
  'status.reclaiming': 'Reopening the room…',
  'status.p2pFailed':
    'Can’t open a direct connection between your two networks. Try another network (a phone hotspot often works), or see the README about TURN servers.',
  'status.roomBusy': 'This room is already open somewhere else (another tab or device?).',
  'status.peerjsLoad': 'Couldn’t load the networking library. Check your connection or ad blocker, then reload.',
  'status.browser': 'Your browser doesn’t support WebRTC, which online games need.',
  'status.version': 'You and your opponent run different versions of the game. Both of you, reload the page.',
  'status.oppOffline': '{name} is disconnected. They can come back with the same link.',
  'rejected.title': 'Can’t join',
  'rejected.full': 'This game already has two players.',
  'rejected.other': 'The host refused the connection.',

  'banner.waitJoin': 'Waiting for your opponent to join…',
  'banner.drawing': '{name} is drawing their maze…',
  'banner.oppReady': '{name} is ready and waiting for you!',
  'banner.meReady': 'You’re ready. Waiting for {name}…',
  'banner.myTurn': 'Your turn: find the way to <b>S</b>!',
  'banner.pending': 'Waiting for the answer…',
  'banner.theirTurn': '{name} is exploring your maze…',
  'banner.won': 'You win! 🎉',
  'banner.lost': '{name} wins this round.',

  'design.title': 'Draw your maze',
  'design.help':
    'Place exactly <b>{walls} walls</b> on the sides of the cells, then the entrance <b class="letter-e">E</b> and the exit <b class="letter-s">S</b>. Your opponent will only see E and S.',
  'design.tools': 'Tools',
  'tool.walls': 'Walls',
  'tool.entrance': 'Entrance',
  'tool.exit': 'Exit',
  'design.random': 'Random',
  'design.clear': 'Clear',
  'design.showPath': 'Show the path',
  'design.counter': '{used} / {total} walls',
  'design.ok': '✓ Valid maze',
  'design.ready': 'I’m ready',
  'design.locked': '🔒 Locked in. No peeking, no changes.',
  'design.budget': 'No walls left: remove one first.',
  'design.hint.walls': 'Click or drag along the sides of the cells to draw or erase walls.',
  'design.hint.entrance': 'Click a cell to place the entrance E.',
  'design.hint.exit': 'Click a cell to place the exit S.',
  'issue.walls-left': ({ count }) => `${count} more ${plural(count, 'wall', 'walls')} to place`,
  'issue.too-many-walls': ({ count }) => `${count} ${plural(count, 'wall', 'walls')} too many`,
  'issue.no-entrance': 'Place the entrance E',
  'issue.no-exit': 'Place the exit S',
  'issue.same-cell': 'E and S must be on different cells',
  'issue.no-path': 'S can’t be reached from E',
  'issue.bad-walls': 'Invalid walls',

  'play.title': '{name}’s maze',
  'play.hint': 'Click a highlighted cell (or use the arrow keys) to try a passage. Click any cell you’ve reached to continue from there.',
  'play.stats': ({ moves, hits }) => `${moves} ${plural(moves, 'move', 'moves')} · ${hits} ${plural(hits, 'wall', 'walls')} hit`,
  'play.resign': 'Concede',
  'play.resignConfirm': 'Concede this round?',
  'play.dpad': 'Directions',
  'dir.U': 'Up',
  'dir.D': 'Down',
  'dir.L': 'Left',
  'dir.R': 'Right',
  'side.yourMaze': 'Your maze: {name}’s progress',
  'log.title': 'Moves',
  'log.empty': 'No moves yet.',
  'log.wall': 'wall',
  'log.exit': 'exit!',

  'over.win': 'You found the exit! 🎉',
  'over.lose': '{name} found your exit first.',
  'over.winResign': '{name} conceded. You win!',
  'over.loseResign': 'You conceded.',
  'over.revealed': '{name}’s maze, revealed',
  'over.legend': 'Dashed walls are the ones you never bumped into.',
  'over.moves': 'Moves',
  'over.hits': 'Walls hit',
  'over.rematch': 'Rematch',
  'over.rematchAccept': '{name} wants a rematch: accept!',
  'over.rematchWaiting': 'Waiting for {name}…',
  'over.menu': 'Back to menu',
  'verify.ok': '✓ Fair play verified: {name}’s answers match the maze they locked in.',
  'verify.wait': 'Waiting for {name} to reveal their maze…',
  'verify.bad': '⚠ Something’s off with {name}’s maze:',
  'verify.hash-mismatch': 'the revealed maze isn’t the one they locked in',
  'verify.rules': 'the maze breaks the rules',
  'verify.es-mismatch': 'E or S moved',
  'verify.lie': 'some answers didn’t match the walls',
  'verify.malformed': 'the revealed data is unreadable',

  'chat.title': 'Chat',
  'chat.placeholder': 'Say something… or bluff',
  'chat.offline': 'Opponent offline',
  'chat.send': 'Send',
  'chat.empty': 'Trash talk welcome.',
  'chat.quick': ['Bonk! 🧱', 'So close…', 'Nope 😏', 'Hmm…', 'GG!'],

  'toast.notYourTurn': 'It’s not your turn.',
  'toast.known': 'You already know that side.',
  'toast.notReached': 'You haven’t reached a neighbouring cell yet.',
  'toast.pickStart': 'First click the cell you want to start from.',
  'toast.wallHit': 'Wall! {name}’s turn.',
  'toast.youStart': 'You start!',
  'toast.theyStart': '{name} starts!',
  'toast.oppJoined': '{name} joined!',
  'toast.oppLeft': '{name} disconnected.',
  'toast.oppReady': '{name} is ready.',
  'toast.rematch': '{name} wants a rematch!',
  'toast.newRoundYou': 'Round {n}: you start.',
  'toast.newRoundThem': 'Round {n}: {name} starts.',
  'title.yourTurn': 'Your turn',
  'title.chat': 'New message',

  'bot.hello': ['Hi! Good luck 🙂', 'Ready to get lost?', 'Beep boop. Let’s play.'],
  'bot.bonk': ['Bonk! 🧱', 'Ouch.', 'Nope!', 'Try again 😏', 'That’s a wall.'],
  'bot.hit': ['Hmm…', 'Interesting.', 'Recalculating…'],
  'bot.win': ['GG! 🤖', 'Found it!', 'Better luck next round.'],
  'bot.lose': ['Well played!', 'GG, you got me.', 'Rematch?'],
};

const fr = {
  tagline: 'Dessine un labyrinthe secret. Trouve la sortie de l’autre avant qu’il trouve la tienne.',
  you: 'Toi',
  opponent: 'Adversaire',
  computer: 'Ordinateur',
  'level.easy': 'facile',
  'level.normal': 'normal',
  'level.hard': 'difficile',
  'nav.menu': 'Menu',
  'nav.sound': 'Son on/off',
  'nav.lang': 'English',

  'home.name': 'Ton prénom',
  'home.namePlaceholder': 'ex. Alex',
  'home.friend.title': 'Jouer avec un ami en ligne',
  'home.friend.desc': 'Crée une partie, envoie le lien, chacun joue sur son écran.',
  'home.size': 'Grille',
  'home.walls': 'Murs chacun',
  'home.wallsHint': '1 – {max}',
  'home.create': 'Créer la partie',
  'home.bot.title': 'Jouer contre l’ordinateur',
  'home.bot.desc': 'Pour s’échauffer contre un robot.',
  'home.level': 'Difficulté',
  'home.start': 'Jouer',
  'home.join.title': 'Rejoindre une partie',
  'home.join.desc': 'Un ami t’a envoyé un lien ou un code ?',
  'home.join.placeholder': 'Code ou lien',
  'home.join.button': 'Rejoindre',
  'home.join.invalid': 'Ce code de partie n’a pas l’air valide.',
  'home.saved.title': 'Tes parties',
  'home.saved.resume': 'Reprendre',
  'home.saved.forget': 'Oublier',
  'home.saved.vs': 'contre {name}',
  'home.saved.waiting': 'en attente d’un adversaire',
  'home.saved.round': 'manche {n}',
  'home.saved.host': 'tu héberges',
  'rules.title': 'Règles du jeu',
  rules: [
    'Chaque joueur dessine en secret un labyrinthe sur sa grille : un nombre exact de <b>murs</b> (un mur = un côté de case), une <b>entrée E</b> et une <b>sortie S</b>.',
    'Quand vous êtes prêts tous les deux, tu découvres seulement où sont le E et le S de l’adversaire.',
    'À ton tour, essaie de passer d’une case déjà atteinte (on part de E) à une case voisine.',
    '<b>Libre ?</b> Tu avances et tu rejoues. <b>Mur ?</b> Il est noté sur ta carte et la main passe.',
    'Tu peux repartir de n’importe quelle case déjà atteinte : cul-de-sac ? Reviens à un embranchement.',
    'Le premier qui atteint le S de l’autre gagne. Le perdant commence la manche suivante.',
    'À la fin, les deux labyrinthes sont révélés et comparés à ce que chacun avait verrouillé : un tricheur se fait prendre.',
  ],

  'join.title': 'Tu es invité !',
  'join.desc': 'Quelqu’un te défie au Labyrinthe. Choisis un prénom et c’est parti.',
  'join.go': 'Rejoindre la partie',

  'lock.title': 'Déjà ouverte',
  'lock.desc': 'Cette partie est ouverte dans un autre onglet de ce navigateur.',
  'lock.here': 'Jouer plutôt ici',
  'lock.away': 'Cette partie est maintenant ouverte dans un autre onglet.',

  'game.room': 'Salle',
  'game.copyLink': 'Copier le lien',
  'game.copied': 'Lien copié !',
  'game.share': 'Partager…',
  'invite.title': 'Invite ton adversaire :',
  'invite.desc': 'envoie-lui ce lien.',
  'invite.meanwhile': 'En attendant, commence à dessiner ton labyrinthe. Il reste secret.',
  'invite.shareText': 'Viens jouer au Labyrinthe avec moi !',

  'status.connecting': 'Connexion à la partie…',
  'status.slow': 'Connexion toujours en cours…',
  'status.hostOffline': 'En attente de {name}…',
  'status.theHost': 'l’hôte',
  'status.network': 'Connexion au serveur perdue, nouvelle tentative…',
  'status.reclaiming': 'Réouverture de la salle…',
  'status.p2pFailed':
    'Impossible d’établir une connexion directe entre vos deux réseaux. Essaie un autre réseau (un partage de connexion marche souvent), ou lis le README à propos des serveurs TURN.',
  'status.roomBusy': 'Cette salle est déjà ouverte ailleurs (autre onglet ou appareil ?).',
  'status.peerjsLoad': 'Impossible de charger la bibliothèque réseau. Vérifie ta connexion ou ton bloqueur de pubs, puis recharge.',
  'status.browser': 'Ton navigateur ne gère pas WebRTC, nécessaire pour jouer en ligne.',
  'status.version': 'Vous n’avez pas la même version du jeu. Rechargez la page tous les deux.',
  'status.oppOffline': '{name} est déconnecté. Il peut revenir avec le même lien.',
  'rejected.title': 'Impossible de rejoindre',
  'rejected.full': 'Cette partie a déjà deux joueurs.',
  'rejected.other': 'L’hôte a refusé la connexion.',

  'banner.waitJoin': 'En attente de ton adversaire…',
  'banner.drawing': '{name} dessine son labyrinthe…',
  'banner.oppReady': '{name} est prêt et t’attend !',
  'banner.meReady': 'Tu es prêt. On attend {name}…',
  'banner.myTurn': 'À toi : trouve le chemin vers <b>S</b> !',
  'banner.pending': 'En attente de la réponse…',
  'banner.theirTurn': '{name} explore ton labyrinthe…',
  'banner.won': 'Gagné ! 🎉',
  'banner.lost': '{name} remporte la manche.',

  'design.title': 'Dessine ton labyrinthe',
  'design.help':
    'Place exactement <b>{walls} murs</b> sur les côtés des cases, puis l’entrée <b class="letter-e">E</b> et la sortie <b class="letter-s">S</b>. Ton adversaire ne verra que E et S.',
  'design.tools': 'Outils',
  'tool.walls': 'Murs',
  'tool.entrance': 'Entrée',
  'tool.exit': 'Sortie',
  'design.random': 'Au hasard',
  'design.clear': 'Effacer',
  'design.showPath': 'Montrer le chemin',
  'design.counter': '{used} / {total} murs',
  'design.ok': '✓ Labyrinthe valide',
  'design.ready': 'Je suis prêt',
  'design.locked': '🔒 Verrouillé. Plus de modifications possibles.',
  'design.budget': 'Plus de murs : enlève-en un d’abord.',
  'design.hint.walls': 'Clique ou glisse le long des côtés des cases pour tracer ou effacer des murs.',
  'design.hint.entrance': 'Clique sur une case pour placer l’entrée E.',
  'design.hint.exit': 'Clique sur une case pour placer la sortie S.',
  'issue.walls-left': ({ count }) => `Encore ${count} ${plural(count, 'mur', 'murs')} à placer`,
  'issue.too-many-walls': ({ count }) => `${count} ${plural(count, 'mur', 'murs')} en trop`,
  'issue.no-entrance': 'Place l’entrée E',
  'issue.no-exit': 'Place la sortie S',
  'issue.same-cell': 'E et S doivent être sur des cases différentes',
  'issue.no-path': 'S n’est pas atteignable depuis E',
  'issue.bad-walls': 'Murs invalides',

  'play.title': 'Le labyrinthe de {name}',
  'play.hint':
    'Clique sur une case en surbrillance (ou utilise les flèches) pour tenter un passage. Clique sur une case déjà atteinte pour repartir de là.',
  'play.stats': ({ moves, hits }) => `${moves} ${plural(moves, 'coup', 'coups')} · ${hits} ${plural(hits, 'mur touché', 'murs touchés')}`,
  'play.resign': 'Abandonner',
  'play.resignConfirm': 'Abandonner cette manche ?',
  'play.dpad': 'Directions',
  'dir.U': 'Haut',
  'dir.D': 'Bas',
  'dir.L': 'Gauche',
  'dir.R': 'Droite',
  'side.yourMaze': 'Ton labyrinthe : la progression de {name}',
  'log.title': 'Coups',
  'log.empty': 'Aucun coup pour l’instant.',
  'log.wall': 'mur',
  'log.exit': 'sortie !',

  'over.win': 'Tu as trouvé la sortie ! 🎉',
  'over.lose': '{name} a trouvé ta sortie en premier.',
  'over.winResign': '{name} a abandonné. Gagné !',
  'over.loseResign': 'Tu as abandonné.',
  'over.revealed': 'Le labyrinthe de {name}, révélé',
  'over.legend': 'Les murs en pointillés sont ceux que tu n’as jamais touchés.',
  'over.moves': 'Coups',
  'over.hits': 'Murs touchés',
  'over.rematch': 'Revanche',
  'over.rematchAccept': '{name} veut une revanche : accepter !',
  'over.rematchWaiting': 'En attente de {name}…',
  'over.menu': 'Retour au menu',
  'verify.ok': '✓ Partie vérifiée : les réponses de {name} correspondent au labyrinthe verrouillé.',
  'verify.wait': 'En attente du labyrinthe de {name}…',
  'verify.bad': '⚠ Quelque chose cloche avec le labyrinthe de {name} :',
  'verify.hash-mismatch': 'le labyrinthe révélé n’est pas celui qui avait été verrouillé',
  'verify.rules': 'le labyrinthe ne respecte pas les règles',
  'verify.es-mismatch': 'E ou S a bougé',
  'verify.lie': 'certaines réponses ne correspondent pas aux murs',
  'verify.malformed': 'les données révélées sont illisibles',

  'chat.title': 'Discussion',
  'chat.placeholder': 'Dis quelque chose… ou bluffe',
  'chat.offline': 'Adversaire hors ligne',
  'chat.send': 'Envoyer',
  'chat.empty': 'Le chambrage est autorisé.',
  'chat.quick': ['Bonk ! 🧱', 'Presque…', 'Raté 😏', 'Hmm…', 'GG !'],

  'toast.notYourTurn': 'Ce n’est pas ton tour.',
  'toast.known': 'Tu connais déjà ce côté.',
  'toast.notReached': 'Tu n’as pas encore atteint de case voisine.',
  'toast.pickStart': 'Clique d’abord sur la case de départ.',
  'toast.wallHit': 'Mur ! Au tour de {name}.',
  'toast.youStart': 'Tu commences !',
  'toast.theyStart': '{name} commence !',
  'toast.oppJoined': '{name} est arrivé !',
  'toast.oppLeft': '{name} s’est déconnecté.',
  'toast.oppReady': '{name} est prêt.',
  'toast.rematch': '{name} veut une revanche !',
  'toast.newRoundYou': 'Manche {n} : tu commences.',
  'toast.newRoundThem': 'Manche {n} : {name} commence.',
  'title.yourTurn': 'À toi de jouer',
  'title.chat': 'Nouveau message',

  'bot.hello': ['Salut ! Bonne chance 🙂', 'Prêt à te perdre ?', 'Bip boup. On joue.'],
  'bot.bonk': ['Bonk ! 🧱', 'Aïe.', 'Raté !', 'Essaie encore 😏', 'C’est un mur.'],
  'bot.hit': ['Hmm…', 'Intéressant.', 'Je recalcule…'],
  'bot.win': ['GG ! 🤖', 'Trouvé !', 'Plus de chance la prochaine fois.'],
  'bot.lose': ['Bien joué !', 'GG, tu m’as eu.', 'Revanche ?'],
};

const DICTS = { en, fr };
export const LANGS = Object.keys(DICTS);

let lang = detect();

function detect() {
  const saved = prefs.get('lang');
  if (LANGS.includes(saved)) return saved;
  const nav = globalThis.navigator?.language?.slice(0, 2);
  return LANGS.includes(nav) ? nav : 'en';
}

export const getLang = () => lang;

export function setLang(next) {
  lang = LANGS.includes(next) ? next : 'en';
  prefs.set('lang', lang);
  document.documentElement.lang = lang;
}

const lookup = (key) => DICTS[lang][key] ?? en[key];
const fill = (s, params) => s.replace(/\{(\w+)\}/g, (_, k) => String(params[k] ?? ''));

/** HTML-safe translation. Arrays (e.g. rules) are returned as-is. */
export function t(key, params = {}) {
  const v = lookup(key);
  if (v === undefined) return esc(key);
  if (Array.isArray(v)) return v;
  const safe = Object.fromEntries(Object.entries(params).map(([k, x]) => [k, typeof x === 'number' ? x : esc(x)]));
  return fill(typeof v === 'function' ? v(safe) : v, safe);
}

/** Plain-text translation (markup stripped). */
export function tt(key, params = {}) {
  const v = lookup(key);
  if (v === undefined) return key;
  if (Array.isArray(v)) return v;
  return fill(typeof v === 'function' ? v(params) : v, params).replace(/<[^>]+>/g, '');
}
