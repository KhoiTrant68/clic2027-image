"""Local reference-assisted blind preference scoring. No third-party web server."""
import argparse
import csv
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
from pathlib import Path
import random
import re
import secrets
import sqlite3
import sys
import time
from urllib.parse import urlparse

from PIL import Image, ImageFilter, ImageOps

QUESTION = 'So với ảnh gốc, bản nào có chất lượng thị giác tốt hơn?'


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(config_path, out):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8-sig'))
    codecs = cfg['codecs']
    if not isinstance(codecs, dict) or len(codecs) < 2 or not all(isinstance(k, str) and k for k in codecs):
        raise ValueError('Cần ít nhất hai codec có tên riêng.')
    folders = {k: (config_path.parent / p).resolve() for k, p in codecs.items()}
    ref = (config_path.parent / cfg['reference']).resolve()
    files = sorted(p.name for p in ref.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'})
    if not files:
        raise ValueError('Không tìm thấy ảnh gốc.')
    # Validate before creating output. Exact filenames and dimensions are deliberate.
    for filename in files:
        with Image.open(ref / filename) as im:
            size = ImageOps.exif_transpose(im).size
        for folder in folders.values():
            with Image.open(folder / filename) as im:
                if ImageOps.exif_transpose(im).size != size:
                    raise ValueError(f'Khác kích thước ảnh: {folder / filename}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'assets').mkdir()
    def asset(path):
        key = secrets.token_hex(24)
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert('RGB')
            # Fresh pixel-only image removes filenames, EXIF, comments and codec metadata.
            clean = Image.frombytes('RGB', image.size, image.tobytes())
            clean.save(out / 'assets' / (key + '.png'))
        return key
    images = []
    for filename in files:
        images.append(dict(source=filename, reference=asset(ref / filename),
                           codecs={name: asset(folder / filename) for name, folder in folders.items()}))
    manifest = dict(version=1, study_id=secrets.token_hex(16), question=cfg.get('question', QUESTION),
                    codecs=list(codecs), images=images)
    save_json(out / 'private_manifest.json', manifest)
    Store(out)
    return manifest


class Store:
    def __init__(self, root, limit=0):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'private_manifest.json').read_text(encoding='utf-8'))
        self.limit = limit
        self.db = self.root / 'ratings.sqlite3'
        with self.connect() as c:
            c.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY, participant TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS trials (
                    id TEXT PRIMARY KEY, token TEXT NOT NULL, position INTEGER NOT NULL,
                    source TEXT NOT NULL, reference TEXT NOT NULL,
                    codec_a TEXT NOT NULL, codec_b TEXT NOT NULL,
                    asset_a TEXT NOT NULL, asset_b TEXT NOT NULL, UNIQUE(token, position));
                CREATE TABLE IF NOT EXISTS votes (
                    trial TEXT PRIMARY KEY, choice TEXT NOT NULL,
                    seconds REAL NOT NULL, created REAL NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.db, timeout=15)
        c.row_factory = sqlite3.Row
        try:
            with c:
                yield c
        finally:
            c.close()

    def start(self):
        token, participant = secrets.token_hex(32), secrets.token_hex(8)
        rng = random.Random(secrets.randbits(128))
        schedule = [(im, pair) for im in self.manifest['images']
                    for pair in itertools.combinations(self.manifest['codecs'], 2)]
        rng.shuffle(schedule)
        if self.limit:
            schedule = schedule[:self.limit]
        with self.connect() as c:
            c.execute('INSERT INTO sessions VALUES (?,?,?)', (token, participant, time.time()))
            for pos, (im, pair) in enumerate(schedule):
                a, b = pair if rng.randrange(2) else pair[::-1]
                c.execute('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?)',
                          (secrets.token_hex(16), token, pos, im['source'], im['reference'],
                           a, b, im['codecs'][a], im['codecs'][b]))
        return token

    def next(self, token):
        with self.connect() as c:
            if not c.execute('SELECT 1 FROM sessions WHERE token=?', (token,)).fetchone():
                raise PermissionError('Phiên chấm không hợp lệ.')
            total = c.execute('SELECT count(*) FROM trials WHERE token=?', (token,)).fetchone()[0]
            row = c.execute('''SELECT t.* FROM trials t LEFT JOIN votes v ON v.trial=t.id
                WHERE t.token=? AND v.trial IS NULL ORDER BY t.position LIMIT 1''', (token,)).fetchone()
        if row is None:
            return dict(done=True, completed=total, total=total)
        return dict(done=False, completed=row['position'], total=total, trial=row['id'],
                    reference='/asset/' + row['reference'] + '.png',
                    a='/asset/' + row['asset_a'] + '.png', b='/asset/' + row['asset_b'] + '.png')

    def vote(self, token, trial, choice, seconds):
        if choice not in {'A', 'B', 'tie', 'skip'}:
            raise ValueError('Lựa chọn không hợp lệ.')
        if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not 0 <= seconds <= 604800:
            raise ValueError('Thời gian chấm không hợp lệ.')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('SELECT token FROM trials WHERE id=?', (trial,)).fetchone()
            if row is None or row['token'] != token:
                raise PermissionError('Trial không thuộc phiên này.')
            previous = c.execute('SELECT choice FROM votes WHERE trial=?', (trial,)).fetchone()
            if previous:
                if previous['choice'] != choice:
                    raise ValueError('Lựa chọn đã được lưu; không thể thay đổi.')
                return  # Idempotent retry after a lost response.
            current = c.execute('''SELECT t.id FROM trials t LEFT JOIN votes v ON v.trial=t.id
                WHERE t.token=? AND v.trial IS NULL ORDER BY t.position LIMIT 1''', (token,)).fetchone()
            if current is None or current['id'] != trial:
                raise ValueError('Trial không phải câu đang chấm.')
            c.execute('INSERT INTO votes VALUES (?,?,?,?)', (trial, choice, seconds, time.time()))


REPO = Path(__file__).resolve().parents[2]
TRACKER = REPO / 'results/tracker/tracker.json'
REPO_DIRS = ('reports', 'plans', 'docs', 'results')  # read-only text files the tracker links to
TEXT_TYPES = {'.md': 'text/markdown', '.txt': 'text/plain', '.json': 'application/json',
              '.csv': 'text/csv', '.log': 'text/plain'}


def repo_file(rel):
    """Resolve a tracker link to a text file under REPO_DIRS, or None (no traversal, no binaries)."""
    path = (REPO / rel).resolve()
    if not any(path.is_relative_to(REPO / d) for d in REPO_DIRS) or path.suffix.lower() not in TEXT_TYPES:
        return None
    return path if path.is_file() else None


def handler(store, tracker=TRACKER):
    page = Path(__file__).with_name('blind_score.html').read_bytes()
    results_page = Path(__file__).with_name('tracker.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No session tokens, asset URLs or filenames in access logs.

        def respond(self, status, data, kind='application/json; charset=utf-8'):
            if isinstance(data, (dict, list)):
                data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.end_headers()
            self.wfile.write(data)

        def token(self):
            return self.headers.get('Authorization', '').removeprefix('Bearer ')

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == '/':
                    if store is None:  # tracker-only server
                        self.send_response(302); self.send_header('Location', '/results'); self.end_headers(); return
                    self.respond(200, page, 'text/html; charset=utf-8')
                elif path == '/results':
                    self.respond(200, results_page, 'text/html; charset=utf-8')
                elif path == '/api/results':
                    if not Path(tracker).is_file():
                        self.respond(404, dict(error='Chưa có dữ liệu thí nghiệm. Chạy: python tools/tracker/build.py'))
                    else:
                        data = json.loads(Path(tracker).read_text(encoding='utf-8'))
                        data['local_files'] = True  # report links open through /repo/ instead of GitHub
                        self.respond(200, data)
                elif path.startswith('/repo/'):
                    from urllib.parse import unquote
                    target = repo_file(unquote(path[len('/repo/'):]))
                    if target is None:
                        self.respond(404, dict(error='Không tìm thấy hoặc không được phép.'))
                    else:
                        self.respond(200, target.read_bytes(), TEXT_TYPES[target.suffix.lower()] + '; charset=utf-8')
                elif store is None:
                    self.respond(404, dict(error='Server này chỉ phục vụ trang kết quả (chạy serve --study để chấm).'))
                elif path == '/api/info':
                    self.respond(200, dict(study=store.manifest['study_id'], question=store.manifest['question']))
                elif path == '/api/next':
                    self.respond(200, store.next(self.token()))
                elif re.fullmatch(r'/asset/[a-f0-9]{48}\.png', path):
                    self.respond(200, (store.root / 'assets' / path.rsplit('/', 1)[1]).read_bytes(), 'image/png')
                else:
                    self.respond(404, dict(error='Không tìm thấy.'))
            except PermissionError as e:
                self.respond(401, dict(error=str(e)))
            except FileNotFoundError:
                self.respond(404, dict(error='Không tìm thấy ảnh.'))

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                self.respond(403, dict(error='Origin không hợp lệ.'))
                return
            if store is None:
                self.respond(404, dict(error='Không có phiên chấm trên server này.'))
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 4096 or not self.headers.get('Content-Type', '').startswith('application/json'):
                    raise ValueError('Body JSON không hợp lệ.')
                body = json.loads(self.rfile.read(length))
                if self.path == '/api/start':
                    self.respond(200, dict(token=store.start()))
                elif self.path == '/api/vote':
                    store.vote(self.token(), body['trial'], body['choice'], body['seconds'])
                    self.respond(200, dict(saved=True))
                else:
                    self.respond(404, dict(error='Không tìm thấy.'))
            except PermissionError as e:
                self.respond(401, dict(error=str(e)))
            except (ValueError, KeyError, TypeError) as e:
                self.respond(400, dict(error=str(e)))
    return Handler


def export(root, out):
    store = Store(root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with store.connect() as c:
        rows = c.execute('''SELECT s.participant,t.source,t.id AS trial,t.codec_a,t.codec_b,
            v.choice,v.seconds,v.created FROM votes v JOIN trials t ON t.id=v.trial
            JOIN sessions s ON s.token=t.token ORDER BY v.created''').fetchall()
    columns = ['participant', 'source', 'trial', 'codec_a', 'codec_b', 'choice', 'seconds', 'created']
    with (out / 'votes.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, columns);writer.writeheader()
        # Prevent spreadsheet formula execution from operator-supplied filenames/names.
        for row in rows:
            writer.writerow({k: ("'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@', '\t', '\r')) else v)
                             for k, v in dict(row).items()})
    counts = {}
    for r in rows:
        a, b = sorted([r['codec_a'], r['codec_b']])
        count = counts.setdefault((a, b), dict(codec_1=a, codec_2=b, wins_1=0, wins_2=0, ties=0, skipped=0))
        if r['choice'] == 'skip': count['skipped'] += 1
        elif r['choice'] == 'tie': count['ties'] += 1
        else:
            winner = r['codec_a'] if r['choice'] == 'A' else r['codec_b']
            count['wins_1' if winner == a else 'wins_2'] += 1
    for c in counts.values():
        n = c['wins_1'] + c['wins_2'] + c['ties']
        c['rated'] = n
        c['preference_share_1'] = (c['wins_1'] + .5 * c['ties']) / n if n else None
    save_json(out / 'summary.json', dict(pairs=list(counts.values()), votes=len(rows),
        note='Thống kê mô tả; tie tính 0.5, skip loại khỏi mẫu số. Không phải kiểm định ý nghĩa thống kê.'))


def demo(root):
    root = Path(root);root.mkdir(parents=True, exist_ok=False)
    for sub in ['reference', 'codec_one', 'codec_two']:
        (root / sub).mkdir()
    import numpy as np
    yy, xx = np.mgrid[:384, :512]
    for i in range(3):
        rgb = np.stack([(xx + i * 30) % 256, (yy * 2) % 256,
                        ((xx // 12 + yy // 12) % 2) * 180 + 40], axis=-1).astype('uint8')
        im = Image.fromarray(rgb)
        name = f'demo-{i+1}.png'
        im.save(root / 'reference' / name)
        im.filter(ImageFilter.GaussianBlur(1 + i * .5)).save(root / 'codec_one' / name)
        im.resize((128, 96)).resize(im.size).save(root / 'codec_two' / name)
    save_json(root / 'config.json', dict(reference='reference',codecs={'demo_blur':'codec_one','demo_resize':'codec_two'}))
    prepare(root / 'config.json', root / 'study')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description='Chấm mù chất lượng ảnh theo cặp A/B.')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare');a.add_argument('--config', type=Path, required=True);a.add_argument('--out', type=Path, required=True)
    a = sub.add_parser('serve');a.add_argument('--study', type=Path, required=True);a.add_argument('--tracker', type=Path, default=TRACKER);a.add_argument('--host', default='127.0.0.1');a.add_argument('--port', type=int, default=8765);a.add_argument('--trials', type=int, default=0)
    a = sub.add_parser('results', help='Chỉ mở trang kết quả thí nghiệm (không chấm).');a.add_argument('--tracker', type=Path, default=TRACKER);a.add_argument('--host', default='127.0.0.1');a.add_argument('--port', type=int, default=8765)
    a = sub.add_parser('export');a.add_argument('--study', type=Path, required=True);a.add_argument('--out', type=Path, required=True)
    a = sub.add_parser('demo');a.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        m = prepare(args.config, args.out);print(f'Đã chuẩn bị {len(m["images"])} ảnh, {len(m["codecs"])} codec.')
    elif args.command == 'demo':
        demo(args.out);print(f'Demo: python tools/blind_score.py serve --study "{args.out / "study"}"')
    elif args.command == 'export':
        export(args.study, args.out);print(f'Đã xuất vào {args.out}')
    else:
        if getattr(args, 'trials', 0) < 0: p.error('--trials phải >= 0')
        store = None if args.command == 'results' else Store(args.study, args.trials)
        server = ThreadingHTTPServer((args.host, args.port), handler(store, args.tracker))
        print(f'Mở http://{args.host}:{server.server_port} — Ctrl+C để dừng.', flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
        finally: server.server_close()


if __name__ == '__main__':
    main()
