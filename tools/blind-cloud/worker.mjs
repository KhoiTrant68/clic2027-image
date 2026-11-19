const codePattern = /^[a-f0-9]{32}$/;
const cookieName = '__Host-clic-blind';
const defaultRound = 'clic30-v2';
const roundPattern = /^[a-z0-9][a-z0-9-]{0,39}$/;
const templates = "(t.participant GLOB 'V2-P[0-9][0-9]' OR t.participant GLOB 'ROUND-*')";
const roundCookie = round => round === defaultRound ? cookieName : `${cookieName}-${round}`;

function json(value, status = 200, headers = {}) {
  return Response.json(value, { status, headers: { 'Cache-Control': 'no-store', ...headers } });
}

async function codeDigest(code) {
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(code));
  return Array.from(new Uint8Array(bytes), byte => byte.toString(16).padStart(2, '0')).join('');
}

async function participant(env, code) {
  if (!codePattern.test(code ?? '')) return null;
  const digest = await codeDigest(code);
  return env.DB.prepare('SELECT participant FROM invitations WHERE digest = ?').bind(digest).first('participant');
}

function cookie(request, name = cookieName) {
  return request.headers.get('Cookie')?.split(';').map(part => part.trim())
    .find(part => part.startsWith(`${name}=`))?.slice(name.length + 1);
}

async function readBody(request) {
  if (!request.headers.get('Content-Type')?.startsWith('application/json')) throw new Error();
  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  let text = '', size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.length;
      if (size > 4096) { await reader.cancel(); throw new Error(); }
      text += decoder.decode(value, { stream: true });
    }
    return JSON.parse(text + decoder.decode());
  } finally {
    reader.releaseLock();
  }
}

const pending = `SELECT t.id, t.reference, t.a, t.b FROM trials t
  LEFT JOIN votes v ON v.trial = t.id
  WHERE t.participant = ? AND v.trial IS NULL ORDER BY t.position LIMIT 1`;

function comparisonPairs(rows) {
  const names = { original: 'MLIC++ gốc', perceptual: 'MLIC++ perceptual', single: 'MLIC++ single', mixed: 'MLIC++ mixed' };
  const pairs = new Map();
  for (const row of rows) {
    const codecs = [row.codec_a, row.codec_b].sort();
    const key = JSON.stringify(codecs);
    if (!pairs.has(key)) pairs.set(key, { methods: codecs.map(codec => names[codec] || codec), count: 0 });
    pairs.get(key).count += row.count;
  }
  return [...pairs.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([, value]) => value);
}

async function methods(env, id) {
  if (!id) return [];
  const rows = await env.DB.prepare('SELECT codec_a, codec_b, count(*) AS count FROM trials WHERE participant = ? GROUP BY codec_a, codec_b').bind(id).all();
  return comparisonPairs(rows.results);
}

async function rounds(request, env) {
  const rows = await env.DB.prepare(`SELECT c.study, c.target_bpp, t.codec_a, t.codec_b, count(*) AS count,
    min(schedule.questions) AS questions
    FROM trials t JOIN trial_context c ON c.trial = t.id
    JOIN (SELECT t.participant, count(*) AS questions FROM trials t WHERE ${templates}
      GROUP BY t.participant) schedule ON schedule.participant = t.participant
    GROUP BY c.study, c.target_bpp, t.codec_a, t.codec_b
    ORDER BY c.study DESC, c.target_bpp, t.codec_a, t.codec_b`).all();
  const studies = Map.groupBy(rows.results, row => row.study);
  return Promise.all([...studies].map(async ([id, entries]) => {
    const participantId = await participant(env, cookie(request, roundCookie(id)));
    const progress = participantId ? await env.DB.prepare(`SELECT count(*) AS total, count(v.trial) AS completed
      FROM trials t LEFT JOIN votes v ON v.trial = t.id WHERE t.participant = ?`).bind(participantId).first() : null;
    return { id, questions: entries[0].questions, rates: [...new Set(entries.map(row => row.target_bpp))],
      pairs: comparisonPairs(entries).map(pair => pair.methods), progress };
  }));
}

async function handle(request, env) {
  const url = new URL(request.url);
  const path = url.pathname;
  if (request.method === 'POST' && request.headers.get('Origin') !== url.origin) {
    return json({ error: 'Nguồn yêu cầu không hợp lệ.' }, 403);
  }
  if (request.method === 'GET' && path === '/results') {
    return env.ASSETS.fetch(request);
  }
  if (request.method === 'GET' && path === '/api/results') {
    const data = await env.ASSETS.fetch(new Request(new URL('/catalog.json', url)));
    return new Response(data.body, { status: data.status,
      headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' } });
  }
  if (request.method === 'POST' && env.STUDY_CLOSED === '1') {
    return json({ error: 'Vòng chấm đã kết thúc.' }, 403);
  }
  if (request.method === 'GET' && path === '/') return env.ASSETS.fetch(request);
  if (request.method === 'GET' && path === '/api/rounds') {
    return json({ rounds: await rounds(request, env), closed: env.STUDY_CLOSED === '1' });
  }
  const round = url.searchParams.get('round') ?? defaultRound;
  if (!roundPattern.test(round)) return json({ error: 'Đợt chấm không hợp lệ.' }, 400);
  const sessionCookie = roundCookie(round);
  if (request.method === 'POST' && path === '/api/start') {
    let code;
    try { code = (await readBody(request)).code?.trim().toLowerCase(); }
    catch { return json({ error: 'Mã mời không hợp lệ.' }, 400); }
    let id;
    if (code === undefined) {
      code = cookie(request, sessionCookie);
      id = await participant(env, code);
      if (!id) {
        const template = await env.DB.prepare(`SELECT t.participant FROM trials t
          JOIN trial_context c ON c.trial = t.id WHERE c.study = ? AND ${templates}
          GROUP BY t.participant ORDER BY random() LIMIT 1`).bind(round).first('participant');
        if (!template) return json({ error: 'Chưa có bộ ảnh để chấm. Hãy thử lại sau.' }, 503);
        code = crypto.randomUUID().replaceAll('-', '');
        id = `ANON-${crypto.randomUUID()}`;
        await env.DB.batch([
          env.DB.prepare('INSERT INTO invitations (digest, participant) VALUES (?, ?)').bind(await codeDigest(code), id),
          env.DB.prepare(`INSERT INTO trials (id, participant, position, source, reference, a, b, codec_a, codec_b)
            SELECT lower(hex(randomblob(16))), ?, position, source, reference, a, b, codec_a, codec_b
            FROM trials WHERE participant = ?`).bind(id, template),
          env.DB.prepare(`INSERT INTO trial_context (trial, study, dataset, target_bpp, image_bytes, pixels, stream_sha256)
            SELECT fresh.id, c.study, c.dataset, c.target_bpp, c.image_bytes, c.pixels, c.stream_sha256
            FROM trials fresh JOIN trials old ON old.position = fresh.position AND old.participant = ?
            JOIN trial_context c ON c.trial = old.id WHERE fresh.participant = ?`).bind(template, id),
        ]);
      }
    } else {
      id = await participant(env, code);
    }
    if (!id) return json({ error: 'Mã mời không hợp lệ.' }, 401);
    return json({ started: true, methods: await methods(env, id) }, 200, {
      'Set-Cookie': `${sessionCookie}=${code}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=2592000`,
    });
  }
  if (request.method === 'GET' && path === '/api/info') {
    const id = await participant(env, cookie(request, sessionCookie));
    return json({ question: 'So với ảnh gốc, bản nào có chất lượng thị giác tốt hơn?',
      resumed: Boolean(id), methods: await methods(env, id) });
  }
  const imagePath = /^\/asset\/[a-f0-9]{48}\.png$/.test(path);
  const isNext = request.method === 'GET' && path === '/api/next';
  const isVote = request.method === 'POST' && path === '/api/vote';
  if (!(isNext || isVote || (imagePath && request.method === 'GET'))) {
    return json({ error: 'Không tìm thấy.' }, 404);
  }
  if (imagePath) {
    const response = await env.ASSETS.fetch(request);
    const headers = new Headers(response.headers);
    headers.set('Cache-Control', 'private, no-store');
    headers.set('X-Content-Type-Options', 'nosniff');
    return new Response(response.body, { status: response.status, headers });
  }
  const id = await participant(env, cookie(request, sessionCookie));
  if (!id) return json({ error: 'Vui lòng bắt đầu đợt chấm.' }, 401);
  if (isNext) {
    const [trial, progress] = await env.DB.batch([
      env.DB.prepare(pending).bind(id),
      env.DB.prepare(`SELECT count(*) AS total, count(v.trial) AS completed FROM trials t
        LEFT JOIN votes v ON v.trial = t.id WHERE t.participant = ?`).bind(id),
    ]);
    const row = trial.results[0];
    const state = { ...progress.results[0], done: !row };
    if (row) Object.assign(state, { trial: row.id, reference: `/asset/${row.reference}.png`,
      a: `/asset/${row.a}.png`, b: `/asset/${row.b}.png` });
    return json(state);
  }
  let vote;
  try { vote = await readBody(request); }
  catch { return json({ error: 'Phiếu không hợp lệ.' }, 400); }
  if (!vote || typeof vote.trial !== 'string' || !codePattern.test(vote.trial) ||
      !['A', 'B', 'tie', 'skip'].includes(vote.choice) || !Number.isFinite(vote.seconds) ||
      vote.seconds < 0 || vote.seconds > 604800) return json({ error: 'Phiếu không hợp lệ.' }, 400);
  await env.DB.prepare(`INSERT INTO votes (trial, choice, seconds, created)
    SELECT current.id, ?, ?, ? FROM (${pending}) current WHERE current.id = ?
    ON CONFLICT(trial) DO NOTHING`).bind(vote.choice, vote.seconds, Date.now(), id, vote.trial).run();
  const stored = await env.DB.prepare(`SELECT v.choice FROM votes v JOIN trials t ON t.id = v.trial
    WHERE t.participant = ? AND t.id = ?`).bind(id, vote.trial).first('choice');
  return stored === vote.choice ? json({ saved: true }) : json({ error: 'Câu này đã đổi hoặc đã được chấm.' }, 409);
}

export default {
  async fetch(request, env) {
    try { return await handle(request, env); }
    catch { return json({ error: 'Máy chủ chưa xử lý được yêu cầu. Hãy thử lại.' }, 503); }
  },
};
