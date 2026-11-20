"""Experiment tracker: recomputed Qhat must reproduce the historical mix; /repo/ links stay inside text dirs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from clic27.blind_score import repo_file  # noqa: E402


class TrackerTest(unittest.TestCase):
    def test_builder_reproduces_reference_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'tracker.json'
            subprocess.run([sys.executable, str(ROOT / 'tools/tracker/build.py'), '--out', str(out)], check=True,
                           capture_output=True)
            data = json.loads(out.read_text(encoding='utf-8'))
        mix = {b['budget']: b['mix']['qhat'] for b in data['budgets']}
        # Historical values from results/2026-10-15_pulse_integration (reports/CLIC2027_pulse_integration_2026-10-15.md).
        for budget, expected in {0.075: 0.1841, 0.15: 0.7891, 0.3: 1.4978}.items():
            self.assertAlmostEqual(mix[budget], expected, places=4)
        for b in data['budgets']:
            self.assertLessEqual(b['mix']['bpp'], b['budget'])
            self.assertEqual(len(b['mix']['choice']), 30)
        points = {p['id']: p for c in data['codecs'] for p in c['points']}
        # Pooled PSNR against the MLIC++ report (reports/CLIC2027_MLICpp_full30_2026-11-02.md: q1 29.07, q4 34.13 dB).
        self.assertAlmostEqual(points['mlicpp/q1']['psnr'], 29.07, places=2)
        self.assertAlmostEqual(points['mlicpp/q4']['psnr'], 34.13, places=2)
        for pid, rows in data['per_image'].items():
            self.assertEqual(len(rows), points[pid]['images'])
            self.assertEqual(points[pid]['partial'], len(rows) < 30)
        # Subset-only codecs never enter a whole-set budget allocation.
        for b in data['budgets']:
            for c in data['codecs']:
                if c['partial']:
                    self.assertIsNone(b['codecs'][c['id']]['alone'])
                    self.assertNotIn(c['id'], b['mix']['used'])

    def test_repo_file_whitelist(self):
        self.assertIsNotNone(repo_file('reports/CLIC2027_ELIC_full30_2026-11-19.md'))
        for bad in ['../pyproject.toml', 'reports/../pyproject.toml', 'src/clic27/blind_score.py',
                    'results/../src/clic27/blind_score.py', '/etc/passwd', 'reports/does-not-exist.md']:
            self.assertIsNone(repo_file(bad), bad)


if __name__ == '__main__':
    unittest.main()
