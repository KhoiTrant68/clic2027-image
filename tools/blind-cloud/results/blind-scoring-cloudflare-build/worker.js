var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// worker.mjs
var codePattern = /^[a-f0-9]{32}$/;
var cookieName = "__Host-clic-blind";
function json(value, status = 200, headers = {}) {
  return Response.json(value, { status, headers: { "Cache-Control": "no-store", ...headers } });
}
__name(json, "json");
async function participant(env, code) {
  if (!codePattern.test(code ?? "")) return null;
  const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code));
  const digest = Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, "0")).join("");
  return env.DB.prepare("SELECT participant FROM invitations WHERE digest = ?").bind(digest).first("participant");
}
__name(participant, "participant");
function cookie(request) {
  return request.headers.get("Cookie")?.split(";").map((part) => part.trim()).find((part) => part.startsWith(`${cookieName}=`))?.slice(cookieName.length + 1);
}
__name(cookie, "cookie");
async function readBody(request) {
  if (!request.headers.get("Content-Type")?.startsWith("application/json")) throw new Error();
  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  let text = "", size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.length;
      if (size > 4096) {
        await reader.cancel();
        throw new Error();
      }
      text += decoder.decode(value, { stream: true });
    }
    return JSON.parse(text + decoder.decode());
  } finally {
    reader.releaseLock();
  }
}
__name(readBody, "readBody");
var pending = `SELECT t.id, t.reference, t.a, t.b FROM trials t
  LEFT JOIN votes v ON v.trial = t.id
  WHERE t.participant = ? AND v.trial IS NULL ORDER BY t.position LIMIT 1`;
async function handle(request, env) {
  const url = new URL(request.url);
  const path = url.pathname;
  if (request.method === "POST" && request.headers.get("Origin") !== url.origin) {
    return json({ error: "Ngu\u1ED3n y\xEAu c\u1EA7u kh\xF4ng h\u1EE3p l\u1EC7." }, 403);
  }
  if (request.method === "GET" && path === "/") return env.ASSETS.fetch(request);
  if (request.method === "POST" && path === "/api/start") {
    let code;
    try {
      code = (await readBody(request)).code?.trim().toLowerCase();
    } catch {
      return json({ error: "M\xE3 m\u1EDDi kh\xF4ng h\u1EE3p l\u1EC7." }, 400);
    }
    if (!await participant(env, code)) return json({ error: "M\xE3 m\u1EDDi kh\xF4ng h\u1EE3p l\u1EC7." }, 401);
    return json({ started: true }, 200, {
      "Set-Cookie": `${cookieName}=${code}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=2592000`
    });
  }
  if (request.method === "GET" && path === "/api/info") {
    return json({
      question: "So v\u1EDBi \u1EA3nh g\u1ED1c, b\u1EA3n n\xE0o c\xF3 ch\u1EA5t l\u01B0\u1EE3ng th\u1ECB gi\xE1c t\u1ED1t h\u01A1n?",
      resumed: Boolean(await participant(env, cookie(request)))
    });
  }
  const imagePath = /^\/asset\/[a-f0-9]{48}\.png$/.test(path);
  const isNext = request.method === "GET" && path === "/api/next";
  const isVote = request.method === "POST" && path === "/api/vote";
  if (!(isNext || isVote || imagePath && request.method === "GET")) {
    return json({ error: "Kh\xF4ng t\xECm th\u1EA5y." }, 404);
  }
  const id = await participant(env, cookie(request));
  if (!id) return json({ error: "Vui l\xF2ng nh\u1EADp m\xE3 m\u1EDDi." }, 401);
  if (imagePath) {
    const response = await env.ASSETS.fetch(request);
    const headers = new Headers(response.headers);
    headers.set("Cache-Control", "private, no-store");
    headers.set("X-Content-Type-Options", "nosniff");
    return new Response(response.body, { status: response.status, headers });
  }
  if (isNext) {
    const [trial, progress] = await env.DB.batch([
      env.DB.prepare(pending).bind(id),
      env.DB.prepare(`SELECT count(*) AS total, count(v.trial) AS completed FROM trials t
        LEFT JOIN votes v ON v.trial = t.id WHERE t.participant = ?`).bind(id)
    ]);
    const row = trial.results[0];
    const state = { ...progress.results[0], done: !row };
    if (row) Object.assign(state, {
      trial: row.id,
      reference: `/asset/${row.reference}.png`,
      a: `/asset/${row.a}.png`,
      b: `/asset/${row.b}.png`
    });
    return json(state);
  }
  let vote;
  try {
    vote = await readBody(request);
  } catch {
    return json({ error: "Phi\u1EBFu kh\xF4ng h\u1EE3p l\u1EC7." }, 400);
  }
  if (!vote || typeof vote.trial !== "string" || !codePattern.test(vote.trial) || !["A", "B", "tie", "skip"].includes(vote.choice) || !Number.isFinite(vote.seconds) || vote.seconds < 0 || vote.seconds > 604800) return json({ error: "Phi\u1EBFu kh\xF4ng h\u1EE3p l\u1EC7." }, 400);
  await env.DB.prepare(`INSERT INTO votes (trial, choice, seconds, created)
    SELECT current.id, ?, ?, ? FROM (${pending}) current WHERE current.id = ?
    ON CONFLICT(trial) DO NOTHING`).bind(vote.choice, vote.seconds, Date.now(), id, vote.trial).run();
  const stored = await env.DB.prepare(`SELECT v.choice FROM votes v JOIN trials t ON t.id = v.trial
    WHERE t.participant = ? AND t.id = ?`).bind(id, vote.trial).first("choice");
  return stored === vote.choice ? json({ saved: true }) : json({ error: "C\xE2u n\xE0y \u0111\xE3 \u0111\u1ED5i ho\u1EB7c \u0111\xE3 \u0111\u01B0\u1EE3c ch\u1EA5m." }, 409);
}
__name(handle, "handle");
var worker_default = {
  async fetch(request, env) {
    try {
      return await handle(request, env);
    } catch {
      return json({ error: "M\xE1y ch\u1EE7 ch\u01B0a x\u1EED l\xFD \u0111\u01B0\u1EE3c y\xEAu c\u1EA7u. H\xE3y th\u1EED l\u1EA1i." }, 503);
    }
  }
};
export {
  worker_default as default
};
//# sourceMappingURL=worker.js.map
