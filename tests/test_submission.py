"""CPU tests of packaging/dispatch only; fixtures do not exercise GPU codecs."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from clic27.submission import check_images, decode, read_package, sha256
from clic27.codec import msillm_stream
sys.path.insert(0, str(ROOT / 'experiments/bakeoff'))
import pulse_codec


class SubmissionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='clic27-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / 'package'
        (self.package / 'streams').mkdir(parents=True)
        self.manifest = {'version': 1, 'images': []}
        self.expected = {'images': []}
        # Two qualities of MS-ILLM exercise grouping without reloading per image.
        for i, (codec, quality) in enumerate([('ditic', 3), ('msillm', 4), ('msillm', 5), ('pulse', 2), ('vtmscc', 32)]):
            name = f'image_{i}'
            image = Image.new('RGB', (32, 24), (i, 23, 81))
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            data = buffer.getvalue()
            if codec == 'msillm':
                data = msillm_stream.pack(quality, (24, 32), (64, 64), data, b'hyper')
            elif codec == 'pulse':
                data = pulse_codec.pack(data)
            (self.package / 'streams' / (name + '.bin')).write_bytes(data)
            self.manifest['images'].append(dict(name=name, width=32, height=24, codec=codec,
                                                quality=quality, sha256=sha256(data)))
            self.expected['images'].append(dict(name=name, width=32, height=24,
                                                pixel_sha256=sha256(image.tobytes())))
        self.save_manifest()
        self.expected_path = self.root / 'expected.json'
        self.expected_path.write_text(json.dumps(self.expected))
        self.runtime = self.root / 'runtime.json'
        self.write_runtime()

    def save_manifest(self):
        (self.package / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def write_runtime(self, mode='ok'):
        command = ['{python}', str(ROOT / 'tests/fixtures/submission_worker.py'),
                   '--input', '{input}', '--output', '{output}', '--metadata', '{metadata}', '--mode', mode]
        self.runtime.write_text(json.dumps({'version': 1, 'workers': dict.fromkeys(
            ['ditic', 'msillm', 'pulse', 'vtmscc'], command)}))

    def zip_package(self, name='bs.zip', extra=None):
        path = self.root / name
        with zipfile.ZipFile(path, 'w') as archive:
            for file in self.package.rglob('*'):
                if file.is_file():
                    archive.write(file, file.relative_to(self.package).as_posix())
            if extra:
                archive.writestr(*extra)
        return path

    def test_bytes_include_metadata_and_zip_overhead(self):
        _, _, directory = read_package(self.package, self.expected)
        archive = self.zip_package()
        _, _, packed = read_package(archive, self.expected)
        self.assertEqual(directory['uncompressed_bytes'], directory['stream_bytes'] + directory['metadata_bytes'])
        self.assertGreater(packed['accounted_bytes'], packed['stream_bytes'])
        self.assertEqual(packed['accounted_bytes'], max(archive.stat().st_size, directory['uncompressed_bytes']))
        self.assertTrue(packed['expected_list_checked'])
        self.assertEqual(packed['pixels'], 5 * 32 * 24)
        # A budget that covers native streams but not their metadata must fail.
        rate = str(directory['stream_bytes'] * 8 / directory['pixels'])
        with self.assertRaisesRegex(ValueError, 'budget exceeded'):
            read_package(self.package, self.expected, rate)
        read_package(archive, self.expected, '100')
        with self.assertRaises(ValueError):
            read_package(archive, self.expected, '0')

    def test_missing_extra_truncated_or_empty_stream(self):
        stream = self.package / 'streams/image_0.bin'
        data = stream.read_bytes()
        for bad in (None, data[:-1], b''):
            with self.subTest(bad=bad):
                if bad is None:
                    stream.unlink()
                else:
                    stream.write_bytes(bad)
                with self.assertRaises(ValueError):
                    read_package(self.package)
                stream.write_bytes(data)
        (self.package / 'unexpected.txt').write_text('extra')
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            read_package(self.package)

    def test_bad_manifest_fields(self):
        for field, value in [('name', '../escape'), ('name', 'CON'), ('name', 'CON.txt'), ('codec', 'unknown'),
                             ('codec', []), ('quality', True), ('quality', 99),
                             ('width', 0), ('height', 1.5), ('sha256', 'bad')]:
            with self.subTest(field=field, value=value):
                old = self.manifest['images'][0][field]
                self.manifest['images'][0][field] = value
                self.save_manifest()
                with self.assertRaises(ValueError):
                    read_package(self.package)
                self.manifest['images'][0][field] = old
        self.manifest['version'] = 2
        self.save_manifest()
        with self.assertRaises(ValueError):
            read_package(self.package)

    def test_duplicate_json_or_image_names(self):
        path = self.package / 'manifest.json'
        path.write_text('{"version":1,"version":1,"images":[]}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            read_package(self.package)
        self.manifest['images'][1]['name'] = self.manifest['images'][0]['name'].upper()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate image'):
            read_package(self.package)

    def test_native_headers_checked_even_with_valid_sha(self):
        self.manifest['images'][1]['quality'] = 5
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'header differs'):
            read_package(self.package)
        self.manifest['images'][1]['quality'] = 4
        for index in (1, 3):
            row = self.manifest['images'][index]
            path = self.package / 'streams' / (row['name'] + '.bin')
            original = path.read_bytes()
            corrupt = b'XX' + original[2:]
            path.write_bytes(corrupt)
            row['sha256'] = sha256(corrupt)
            self.save_manifest()
            with self.assertRaises(ValueError):
                read_package(self.package)
            path.write_bytes(original)
            row['sha256'] = sha256(original)
        self.save_manifest()

    def test_archive_traversal_duplicate_and_symlink(self):
        for i, extra in enumerate([('../escaped.bin', b'x'), ('streams/../escaped.bin', b'x'),
                                   ('MANIFEST.JSON', b'{}')]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                read_package(self.zip_package(f'bad{i}.zip', extra))
        archive = self.zip_package()
        info = zipfile.ZipInfo('link')
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        with zipfile.ZipFile(archive, 'a') as handle:
            handle.writestr(info, '../other')
        with self.assertRaisesRegex(ValueError, 'Invalid ZIP member'):
            read_package(archive)
        self.assertFalse((self.root / 'escaped.bin').exists())

    def test_independent_list_catches_missing_images_and_wrong_dimensions(self):
        wrong = json.loads(json.dumps(self.expected))
        wrong['images'].pop()
        with self.assertRaisesRegex(ValueError, 'Image list differs'):
            read_package(self.package, wrong)
        wrong = json.loads(json.dumps(self.expected))
        wrong['images'][0]['width'] += 1
        with self.assertRaisesRegex(ValueError, 'Dimensions differ'):
            read_package(self.package, wrong)

    def test_dispatch_all_codecs_in_fresh_processes(self):
        output = self.root / 'images'
        report = decode(self.zip_package(), output, self.runtime, self.expected)
        self.assertEqual(len(report['batches']), 4)
        self.assertEqual(next(r['images'] for r in report['batches'] if r['codec'] == 'msillm'), 2)
        self.assertEqual(len(check_images(output, self.expected['images'])), 5)
        self.assertTrue(all(r['pixel_hash_checked'] for r in report['output']))
        with self.assertRaises(FileExistsError):
            decode(self.package, output, self.runtime)

    def test_worker_failures_do_not_publish_partial_output(self):
        for mode in ['fail', 'missing', 'extra', 'wrong-size', 'wrong-mode', 'wrong-pixels', 'invalid-png', 'timeout']:
            with self.subTest(mode=mode):
                self.write_runtime(mode)
                output = self.root / mode
                with self.assertRaises((OSError, ValueError, RuntimeError)):
                    decode(self.package, output, self.runtime, self.expected,
                           timeout=.1 if mode == 'timeout' else 10)
                self.assertFalse(output.exists())

    def test_late_worker_failure_does_not_publish_earlier_results(self):
        config = json.loads(self.runtime.read_text())
        config['workers']['vtmscc'][-1] = 'fail'
        self.runtime.write_text(json.dumps(config))
        output = self.root / 'output'
        with self.assertRaisesRegex(RuntimeError, 'intentional worker failure'):
            decode(self.package, output, self.runtime)
        self.assertFalse(output.exists())

    def test_missing_worker_and_unknown_placeholder_fail_before_dispatch(self):
        for workers in [{}, {'ditic': ['{python}', '{not_a_placeholder}']}]:
            if workers:
                command = workers['ditic']
                workers = dict.fromkeys(['ditic', 'pulse', 'vtmscc', 'msillm'], command)
            self.runtime.write_text(json.dumps({'version': 1, 'workers': workers}))
            with self.assertRaises(ValueError):
                decode(self.package, self.root / 'output', self.runtime)
        self.assertFalse((self.root / 'output').exists())

    def test_cli_preflight_and_decode_and_exit_codes(self):
        archive = self.zip_package()
        preflight = [sys.executable, str(ROOT / 'clic/tools/check_submission.py'), '--input', str(archive),
                     '--expected', str(self.expected_path), '--budget-bpp', '100']
        result = subprocess.run(preflight, capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)['decode_executed'])
        output, report = self.root / 'cli-images', self.root / 'check.json'
        result = subprocess.run(preflight + ['--runtime', str(self.runtime), '--output', str(output),
                                            '--report', str(report)], capture_output=True, text=True, check=True)
        self.assertEqual(len(json.loads(result.stdout)['output']), 5)
        self.assertTrue(report.is_file())
        result = subprocess.run(preflight + ['--output', str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(preflight + ['--report', str(report)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        for bad_report in [output / 'report.json', self.package / 'report.json']:
            command = preflight + ['--output', str(output), '--report', str(bad_report)]
            # For input immutability, inspect a directory package rather than its ZIP.
            if bad_report.parent == self.package:
                command[command.index(str(archive))] = str(self.package)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(bad_report.exists())
        result = subprocess.run([sys.executable, str(ROOT / 'clic/submission/decode.py'),
                                 '--input', str(archive), '--output', str(self.root / 'entry-images'),
                                 '--runtime', str(self.runtime)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(preflight[:-1] + ['0.0001'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('budget exceeded', result.stderr)
        self.expected_path.write_text('null')
        result = subprocess.run(preflight, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('images list', result.stderr)


if __name__ == '__main__':
    unittest.main()
