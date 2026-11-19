"""Blindness, persistence and answer integrity through the actual HTTP API."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from clic27.blind_score import prepare, Store, handler, export


class BlindScoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, color in [('reference', 120), ('secret_codec_x', 110), ('secret_codec_y', 130)]:
            folder = self.root / name;folder.mkdir()
            for index in range(2):
                Image.new('RGB', (32, 24), (color, index, 50)).save(folder / f'source-secret-{index}.png')
        cfg = dict(reference='reference', codecs={n: n for n in ['secret_codec_x', 'secret_codec_y']})
        (self.root / 'config.json').write_text(json.dumps(cfg))
        self.study = self.root / 'study'
        prepare(self.root / 'config.json', self.study)
        self.store = Store(self.study)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.store))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True);self.thread.start()
        self.addCleanup(self.close_server)
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def close_server(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def api(self, path, token='', body=None, origin=None):
        headers = {'Authorization': 'Bearer ' + token}
        if body is not None: headers['Content-Type'] = 'application/json'
        if origin: headers['Origin'] = origin
        req = Request(self.base + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
        with urlopen(req) as r:
            return json.load(r)

    def test_blind_api_and_private_files(self):
        token = self.api('/api/start', body={})['token']
        trial = self.api('/api/next', token)
        serialized = json.dumps(trial)
        for secret in ['secret_codec_x', 'secret_codec_y', 'source-secret']:
            self.assertNotIn(secret, serialized)
        with urlopen(self.base + trial['a']) as r:
            with Image.open(r) as image:
                self.assertEqual(image.size, (32, 24));self.assertEqual(image.info, {})
        for path in ['/private_manifest.json', '/ratings.sqlite3', '/asset/../private_manifest.json']:
            with self.assertRaises(HTTPError) as ctx: urlopen(self.base + path)
            self.assertEqual(ctx.exception.code, 404)

    def test_vote_resume_idempotency_ownership_and_export(self):
        token = self.api('/api/start', body={})['token']
        other = self.api('/api/start', body={})['token']
        first = self.api('/api/next', token)
        payload = dict(trial=first['trial'], choice='A', seconds=2.5)
        with self.assertRaises(HTTPError) as ctx: self.api('/api/vote', other, payload)
        self.assertEqual(ctx.exception.code, 401)
        self.api('/api/vote', token, payload);self.api('/api/vote', token, payload)
        second = Store(self.study).next(token)
        self.assertEqual(second['completed'], 1)
        with self.assertRaises(HTTPError): self.api('/api/vote', token, dict(payload, choice='B'))
        self.api('/api/vote', token, dict(trial=second['trial'], choice='tie', seconds=3))
        self.assertTrue(self.api('/api/next', token)['done'])
        output = self.root / 'export';export(self.study, output)
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        pair = summary['pairs'][0]
        self.assertEqual(summary['votes'], 2);self.assertEqual(pair['ties'], 1)
        with self.store.connect() as c:
            codec_a = c.execute('SELECT codec_a FROM trials WHERE id=?', (first['trial'],)).fetchone()[0]
        self.assertEqual(pair['wins_1' if codec_a == pair['codec_1'] else 'wins_2'], 1)
        self.assertNotIn(token, (output / 'votes.csv').read_text(encoding='utf-8-sig'))

    def test_cross_origin_and_invalid_choice_rejected(self):
        with self.assertRaises(HTTPError) as ctx:
            self.api('/api/start', body={}, origin='https://unrelated.invalid')
        self.assertEqual(ctx.exception.code, 403)
        token = self.store.start();t = self.store.next(token)
        with self.assertRaises(ValueError): self.store.vote(token, t['trial'], 'codec_x', 2)
        with self.assertRaises(ValueError): self.store.vote(token, t['trial'], 'A', float('nan'))

    def test_missing_or_misaligned_image_fails_before_output(self):
        Image.new('RGB', (16, 24)).save(self.root / 'secret_codec_x' / 'source-secret-0.png')
        with self.assertRaises(ValueError): prepare(self.root / 'config.json', self.root / 'bad')
        self.assertFalse((self.root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
