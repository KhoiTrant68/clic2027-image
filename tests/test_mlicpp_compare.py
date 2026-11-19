"""Allocator/report regression with historical metric fixtures, not MLIC++ quality evidence."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MlicComparisonTest(unittest.TestCase):
    def test_complete_grid_and_no_degradation_of_frozen_baseline(self):
        base = ROOT / 'results/2026-10-15_pulse_integration/integrated_bakeoff'
        with (base / 'allocation.csv').open() as handle:
            allocation = [r for r in csv.DictReader(handle) if r['cand'] == 'mix' and float(r['rate']) == .075]
        with (base / 'metrics.csv').open() as handle:
            metrics = {(r['cand'], r['key'], r['name']): r for r in csv.DictReader(handle)}
        sizes = json.loads((base / 'budgets.json').read_text())['sizes']
        with tempfile.TemporaryDirectory(prefix='mlic-compare-test-') as temp:
            out = Path(temp)
            manifest = dict(qualities=[1], images=[dict(name=r['name'], height=sizes[r['name']][0], width=sizes[r['name']][1])
                                                  for r in allocation],
                            qhat_sha256=hashlib.sha256((ROOT / 'results/2026-10-15_new_codecs/qhat_v0.json').read_bytes()).hexdigest())
            (out / 'manifest.json').write_text(json.dumps(manifest))
            status = dict(state='complete', fresh_pixel_parity=True, metrics=True)
            (out / 'status.json').write_text(json.dumps(status))
            rows = []
            for row in allocation:
                metric = metrics[row['point_cand'], row['key'], row['name']]
                rows.append(dict(quality=1, name=row['name'], bytes=row['bytes'], qhat_v0=row['score'],
                                 **{k: metric[k] for k in ['mse', 'msssim', 'lpips_alex', 'lpips_vgg', 'dists', 'clipiqa']}))
            with (out / 'metrics.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            command = [sys.executable, str(ROOT / 'experiments/mlicpp/compare.py'), '--out', str(out)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((out / 'comparison.json').read_text())
            expanded = [r for r in report['results'] if r['method'] == 'pulse_mix_plus_mlicpp']
            self.assertEqual(len(expanded), 3)
            self.assertTrue(all(r['delta_qhat_vs_frozen'] >= 0 for r in expanded))
            self.assertTrue(all(r['delta_qhat_vs_reoptimized'] >= 0 for r in expanded))
            self.assertTrue(all(r['bytes'] <= int(r['budget_bpp'] * 86245376 / 8) for r in expanded))
            with (out / 'metrics.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows[:-1])
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('Incomplete', run.stderr)
            status['fresh_pixel_parity'] = False
            (out / 'status.json').write_text(json.dumps(status))
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('parity', run.stderr)


if __name__ == '__main__':
    unittest.main()
