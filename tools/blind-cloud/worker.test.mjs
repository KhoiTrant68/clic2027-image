import assert from 'node:assert/strict';
import worker from './worker.mjs';
import { readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';

const base = 'https://example.test';
let rows = [{ codec_a: 'original', codec_b: 'mixed', count: 3 },
  { codec_a: 'mixed', codec_b: 'original', count: 2 }];
const env = {
  DB: { prepare(sql) { return { bind() { return this; },
    async first() { return 'P01'; }, async all() { return { results: rows }; } }; } },
  ASSETS: { async fetch() { return Response.json({ public: true }); } },
};
const info = () => worker.fetch(new Request(`${base}/api/info`, { headers: { Cookie: '__Host-clic-blind='+'a'.repeat(32) } }), env).then(r=>r.json());
const first = await info();
assert.deepEqual(first.methods, [{ methods: ['MLIC++ mixed', 'MLIC++ gốc'], count: 5 }]);
rows = rows.map(row=>({ ...row, codec_a:row.codec_b, codec_b:row.codec_a }));
assert.deepEqual((await info()).methods, first.methods, 'Pair overview must not reveal A/B orientation');
const anonymous = await worker.fetch(new Request(`${base}/api/info`),env);
assert.deepEqual((await anonymous.json()).methods,[]);
for (const path of ['/api/results','/asset/'+'b'.repeat(48)+'.png']) {
  assert.equal((await worker.fetch(new Request(base+path),env)).status,200);
}
assert.equal((await worker.fetch(new Request(`${base}/api/next`),env)).status,401);
console.log('Public results/images and unordered method overview verified; votes require a browser session.');

let statements, storedDigest, storedId, batches = 0;
const anonymousEnv = { ...env, DB: {
  prepare(sql) { return { sql, args: [], bind(...args) { this.args = args; return this; },
    async first() { return sql.includes('GLOB') ? 'V2-P01' : this.args[0] === storedDigest ? storedId : null; },
    async all() { return { results: rows }; } }; },
  async batch(items) { statements = items; batches++; storedDigest = items[0].args[0]; storedId = items[0].args[1]; },
} };
async function start(cookie) {
  return worker.fetch(new Request(base+'/api/start', { method:'POST',
    headers: { Origin:base, 'Content-Type':'application/json', ...(cookie ? { Cookie:cookie } : {}) }, body:'{}' }), anonymousEnv);
}
const created = await start();
assert.equal(created.status,200);
assert.equal(statements.length,3);
assert.match(storedId,/^ANON-/);
assert.ok((await created.json()).methods.length);
const cookieHeader = created.headers.get('set-cookie').split(';')[0];
assert.equal((await start(cookieHeader)).status,200);
assert.equal(batches,1,'Returning browser must retain the same participant and votes');
console.log('Anonymous session creation and same-browser resumption verified.');

const db = new DatabaseSync(':memory:');
db.exec(readFileSync(new URL('./schema.sql', import.meta.url), 'utf8'));
db.exec(`CREATE TABLE trial_context (
  trial TEXT PRIMARY KEY REFERENCES trials(id), study TEXT NOT NULL, dataset TEXT NOT NULL,
  target_bpp REAL NOT NULL, image_bytes INTEGER NOT NULL, pixels INTEGER NOT NULL, stream_sha256 TEXT NOT NULL
)`);
for (const [index, study, id, candidate] of [
  [1, 'clic30-v2', 'V2-P01', 'mixed'],
  [2, 'clic30-v3', 'ROUND-clic30-v3-P01', 'mixed-v3'],
]) {
  const trial = String(index).repeat(32);
  db.prepare('INSERT INTO invitations VALUES (?, ?)').run(id, id);
  db.prepare('INSERT INTO trials VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?)').run(
    trial, id, 'img.png', '0'.repeat(48), 'a'.repeat(48), 'b'.repeat(48), candidate, 'original');
  db.prepare('INSERT INTO trial_context VALUES (?, ?, ?, 0.075, 100, 1000, ?)').run(trial, study, 'clic30', 'hash');
}
const d1 = {
  prepare(sql) {
    const query = db.prepare(sql);
    return {
      args: [], bind(...args) { this.args = args; return this; },
      all() { return { results: query.all(...this.args) }; },
      first(column) {
        const row = query.get(...this.args);
        return column ? row?.[column] : row;
      },
      run() { return query.run(...this.args); },
    };
  },
  batch(items) { return items.map(item => item.all()); },
};
async function call(path, { cookies, body } = {}) {
  return worker.fetch(new Request(base + path, {
    method: body ? 'POST' : 'GET',
    headers: { Origin: base, 'Content-Type': 'application/json', ...(cookies ? { Cookie: cookies } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  }), { ...env, DB: d1 });
}
const initial = await (await call('/api/rounds')).json();
assert.deepEqual(initial.rounds.map(({ id, questions }) => [id, questions]), [['clic30-v3', 1], ['clic30-v2', 1]]);
assert.deepEqual(initial.rounds[0].pairs, [['mixed-v3', 'MLIC++ gốc']]);
assert.equal((await call('/api/start?round=missing', { body: {} })).status, 503);
const oldCookie = (await call('/api/start', { body: {} })).headers.get('Set-Cookie').split(';')[0];
const newCookie = (await call('/api/start?round=clic30-v3', { cookies: oldCookie, body: {} })).headers.get('Set-Cookie').split(';')[0];
assert.match(newCookie, /^__Host-clic-blind-clic30-v3=/);
const cookies = oldCookie + '; ' + newCookie;
const next = await (await call('/api/next?round=clic30-v3', { cookies })).json();
assert.deepEqual(Object.keys(next).sort(), ['a', 'b', 'completed', 'done', 'reference', 'total', 'trial']);
const vote = { trial: next.trial, choice: 'A', seconds: 1 };
assert.equal((await call('/api/vote', { cookies, body: vote })).status, 409);
assert.equal((await call('/api/vote?round=clic30-v3', { cookies, body: vote })).status, 200);
const updated = await (await call('/api/rounds', { cookies })).json();
assert.deepEqual(updated.rounds.map(({ progress }) => progress.completed), [1, 0]);
console.log('Two rounds on one browser maintain separate ballots, progress and anonymous A/B.');
