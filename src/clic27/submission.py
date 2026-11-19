"""CPU package validation and batch dispatch; codec workers are packaged separately."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import string
import subprocess
import sys
import tempfile
import time
import zipfile

from clic27.codec import msillm_stream

CODECS = {'ditic': range(5), 'msillm': range(1, 7), 'pulse': range(5), 'vtmscc': range(64)}
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')
SHA = re.compile(r'[0-9a-f]{64}\Z')
RESERVED_NAMES = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{prefix}{i}' for prefix in ('COM', 'LPT') for i in range(1, 10)}
# Internal limits for the local package format, not competition rules.
MAX_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON field: {key}')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def image_records(document, transmitted=False):
    if not isinstance(document, dict) or not isinstance(document.get('images'), list) or not document['images']:
        raise ValueError('Expected a non-empty images list')
    if transmitted and (type(document.get('version')) is not int or document['version'] != 1
                        or set(document) != {'version', 'images'}):
        raise ValueError('Unknown package version or manifest fields')
    records, names = [], set()
    for row in document['images']:
        if not isinstance(row, dict):
            raise ValueError('Each image record must be an object')
        name = row.get('name')
        if (not isinstance(name, str) or not NAME.fullmatch(name) or name.casefold() in names
                or name.upper() in RESERVED_NAMES):
            raise ValueError(f'Unsafe or duplicate image name: {name!r}')
        names.add(name.casefold())
        w, h = row.get('width'), row.get('height')
        if (type(w) is not int or type(h) is not int or not (0 < w <= 65536 and 0 < h <= 65536)
                or w * h > 100_000_000):
            raise ValueError(f'Invalid image dimensions: {name}')
        if transmitted:
            if set(row) != {'name', 'width', 'height', 'codec', 'quality', 'sha256'}:
                raise ValueError(f'Unknown or missing image fields: {name}')
            codec, quality = row['codec'], row['quality']
            if (not isinstance(codec, str) or codec not in CODECS or type(quality) is not int
                    or quality not in CODECS[codec]):
                raise ValueError(f'Invalid codec/quality: {name}')
            if not isinstance(row['sha256'], str) or not SHA.fullmatch(row['sha256']):
                raise ValueError(f'Invalid stream SHA-256: {name}')
        elif 'pixel_sha256' in row:
            if not isinstance(row['pixel_sha256'], str) or not SHA.fullmatch(row['pixel_sha256']):
                raise ValueError(f'Invalid pixel SHA-256: {name}')
        records.append(row)
    return records


def read_package(source, expected=None, budget_bpp=None):
    """Read flat streams plus manifest, checking inventory, hashes and conservative bytes."""
    source = Path(source)
    archive_bytes = None
    if source.is_dir():
        files = {}
        for path in source.rglob('*'):
            if path.is_symlink():
                raise ValueError(f'Symlinks are not package files: {path}')
            name = path.relative_to(source).as_posix()
            if path.is_dir():
                if name != 'streams':
                    raise ValueError(f'Unexpected package directory: {name}')
            elif path.is_file():
                files[name] = path
            else:
                raise ValueError(f'Not a regular package file: {name}')
        sizes = {name: path.stat().st_size for name, path in files.items()}
        if sum(sizes.values()) > MAX_BYTES or sizes.get('manifest.json', 0) > MAX_MANIFEST_BYTES:
            raise ValueError('Package exceeds internal size limits')
        blobs = {name: path.read_bytes() for name, path in files.items()}
    else:
        archive_bytes = source.stat().st_size
        if archive_bytes > MAX_BYTES:
            raise ValueError('Archive exceeds internal size limit')
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            names = [item.filename.casefold().rstrip('/') for item in infos]
            if len(set(names)) != len(names):
                raise ValueError('Duplicate ZIP members (including case collisions)')
            if sum(item.file_size for item in infos) > MAX_BYTES:
                raise ValueError('Inflated ZIP exceeds internal size limit')
            for item in infos:
                if stat.S_ISLNK(item.external_attr >> 16) or (item.is_dir() and item.filename != 'streams/'):
                    raise ValueError(f'Invalid ZIP member: {item.filename}')
                if item.filename == 'manifest.json' and item.file_size > MAX_MANIFEST_BYTES:
                    raise ValueError('Manifest exceeds internal size limit')
            # Read members directly; never extract paths supplied by the archive.
            blobs = {item.filename: archive.read(item) for item in infos if not item.is_dir()}
    if 'manifest.json' not in blobs:
        raise ValueError('Missing manifest.json')
    manifest = load_json(blobs['manifest.json'])
    records = image_records(manifest, transmitted=True)
    inventory = {'manifest.json'} | {f"streams/{row['name']}.bin" for row in records}
    if set(blobs) != inventory:
        raise ValueError(f'Package inventory mismatch: missing={sorted(inventory-set(blobs))}, extra={sorted(set(blobs)-inventory)}')
    for row in records:
        data = blobs[f"streams/{row['name']}.bin"]
        if not data or sha256(data) != row['sha256']:
            raise ValueError(f"Empty or checksum-mismatched stream: {row['name']}")
        if row['codec'] == 'msillm':
            quality, size, _, _, _ = msillm_stream.unpack(data)
            if quality != row['quality'] or size != (row['height'], row['width']):
                raise ValueError(f"MS-ILLM header differs from manifest: {row['name']}")
        elif row['codec'] == 'pulse':
            # Same outer framing as experiments/bakeoff/pulse_codec.py; keep its
            # standalone deployment independent of this local harness.
            if len(data) <= 4 or data[:4] != b'PL\x01\x00':
                raise ValueError(f"Invalid PULSE framing: {row['name']}")
    if expected is not None:
        reference = {r['name']: r for r in image_records(expected)}
        if {r['name'] for r in records} != set(reference):
            raise ValueError('Image list differs from independent expected list')
        for row in records:
            ref = reference[row['name']]
            if (row['width'], row['height']) != (ref['width'], ref['height']):
                raise ValueError(f"Dimensions differ from expected list: {row['name']}")
    total = sum(map(len, blobs.values()))
    pixels = sum(row['width'] * row['height'] for row in records)
    accounted = max(total, archive_bytes or 0)
    summary = dict(images=len(records), pixels=pixels, stream_bytes=total-len(blobs['manifest.json']),
                   metadata_bytes=len(blobs['manifest.json']), uncompressed_bytes=total,
                   archive_bytes=archive_bytes, accounted_bytes=accounted,
                   actual_bpp=8 * accounted / pixels,
                   accounting='max(uncompressed files including manifest, ZIP bytes)',
                   expected_list_checked=expected is not None,
                   native_header_checks={codec: {'msillm': 'framing, quality, dimensions',
                                                'pulse': 'four-byte framing only'}.get(codec, 'not implemented')
                                         for codec in sorted({r['codec'] for r in records})},
                   manifest_sha256=sha256(blobs['manifest.json']))
    if budget_bpp is not None:
        rate = Fraction(str(budget_bpp))
        if rate <= 0:
            raise ValueError('Budget bpp must be positive')
        limit = rate * pixels / 8
        summary['budget_bytes'] = limit.numerator // limit.denominator
        if accounted > summary['budget_bytes']:
            raise ValueError(f"Byte budget exceeded: {accounted} > {summary['budget_bytes']} (including metadata/ZIP)")
    return manifest, blobs, summary


def check_images(directory, records):
    """Verify exact file inventory, PNG decoding, dimensions, RGB and optional pixel hash."""
    from PIL import Image
    directory = Path(directory)
    wanted = {r['name'] + '.png' for r in records}
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != wanted:
        raise ValueError('Decoded image inventory differs from expected list')
    result = []
    for row in records:
        path = directory / (row['name'] + '.png')
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Not a regular output image: {path.name}')
        with Image.open(path) as image:
            if (image.format != 'PNG' or image.mode != 'RGB' or image.n_frames != 1
                    or image.size != (row['width'], row['height'])):
                raise ValueError(f'Invalid PNG format/mode/dimensions: {path.name}')
            image.load()
            pixel_hash = sha256(image.tobytes())
        if 'pixel_sha256' in row and pixel_hash != row['pixel_sha256']:
            raise ValueError(f'Pixel checksum mismatch: {path.name}')
        result.append(dict(name=row['name'], width=row['width'], height=row['height'],
                           pixel_sha256=pixel_hash, pixel_hash_checked='pixel_sha256' in row))
    return result


def decode(source, output, runtime_path, expected=None, budget_bpp=None, timeout=300):
    """Run one fresh worker per codec, then publish only a fully checked image set."""
    started = time.perf_counter()
    manifest, blobs, summary = read_package(source, expected, budget_bpp)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite output: {output}')
    source_path = Path(source).resolve()
    if source_path.is_dir() and output.is_relative_to(source_path):
        raise ValueError('Output must be outside the input package')
    runtime_path = Path(runtime_path).resolve()
    runtime = load_json(runtime_path.read_bytes())
    if not isinstance(runtime, dict) or type(runtime.get('version')) is not int or runtime['version'] != 1:
        raise ValueError('Unsupported runtime config version')
    workers = runtime.get('workers')
    codecs = sorted({r['codec'] for r in manifest['images']})
    if not isinstance(workers, dict) or any(codec not in workers for codec in codecs):
        raise ValueError(f'Missing workers for package codecs: {codecs}')
    for codec in codecs:
        command = workers[codec]
        if not isinstance(command, list) or not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError(f'Worker must be a non-empty argv list: {codec}')
        for arg in command:
            for _, key, spec, conversion in string.Formatter().parse(arg):
                if key is not None and (key not in {'python', 'work', 'input', 'output', 'metadata', 'report'} or spec or conversion):
                    raise ValueError(f'Unknown worker placeholder: {key}')
    if not isinstance(timeout, (int, float)) or not (0 < timeout < float('inf')):
        raise ValueError('Worker timeout must be positive and finite')
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    batches = []
    with tempfile.TemporaryDirectory(prefix='clic27-mix-') as temp:
        root = Path(temp)
        images = root / 'images'
        images.mkdir()
        for codec in codecs:
            rows = [r for r in manifest['images'] if r['codec'] == codec]
            work = root / codec
            streams, decoded = work / 'streams', work / 'decoded'
            streams.mkdir(parents=True)
            for row in rows:
                (streams / (row['name'] + '.bin')).write_bytes(blobs[f"streams/{row['name']}.bin"])
            metadata = work / 'selection.json'
            metadata.write_text(json.dumps({'version': 1, 'images': rows}), encoding='utf-8')
            values = dict(python=sys.executable, work=str(work), input=str(streams), output=str(decoded),
                          metadata=str(metadata), report=str(work / 'report.json'))
            argv = [arg.format_map(values) for arg in workers[codec]]
            t0 = time.perf_counter()
            log = work / 'worker.log'
            with log.open('wb') as handle:
                try:
                    subprocess.run(argv, cwd=runtime_path.parent, env=env, timeout=timeout,
                                   stdout=handle, stderr=subprocess.STDOUT, check=True)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    # Keep bounded diagnostics even though temporary files are removed.
                    handle.flush()
                    with log.open('rb') as reader:
                        reader.seek(max(0, log.stat().st_size - 4000))
                        tail = reader.read().decode('utf-8', errors='replace')
                    raise RuntimeError(f'{codec} worker failed: {exc}\n{tail}') from exc
            seconds = time.perf_counter() - t0
            checks = check_images(decoded, rows)
            for row in rows:
                shutil.copyfile(decoded / (row['name'] + '.png'), images / (row['name'] + '.png'))
            batches.append(dict(codec=codec, images=len(rows), process_wall_s=seconds, output=checks))
        output_checks = check_images(images, image_records(expected) if expected is not None else manifest['images'])
        output.parent.mkdir(parents=True, exist_ok=True)
        # No output directory is created until every worker and image has passed.
        output.mkdir()
        try:
            for path in images.iterdir():
                shutil.copyfile(path, output / path.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return dict(package=summary, batches=batches, output=output_checks,
                total_wall_s=time.perf_counter()-started,
                network_isolation='not enforced; use container --network none for deployment',
                scope='package and output checks; no GPU parity or quality claim')


def main(check=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='bs.zip or unpacked package directory')
    parser.add_argument('--expected', type=Path, required=check, help='independent image list, outside bs.zip')
    parser.add_argument('--budget-bpp', help='positive bpp ceiling; includes manifest and ZIP overhead')
    parser.add_argument('--runtime', type=Path, help='trusted decoder worker argv config')
    parser.add_argument('--output', type=Path, help='new output directory when decoding; existing images when only checking')
    parser.add_argument('--timeout', type=float, default=300, help='seconds per worker')
    parser.add_argument('--report', type=Path, help='new JSON report file; never overwrite')
    args = parser.parse_args()
    try:
        if args.report is not None:
            report_path = args.report.resolve()
            if (report_path.exists() or (args.output and report_path.is_relative_to(args.output.resolve()))
                    or (args.input.is_dir() and report_path.is_relative_to(args.input.resolve()))):
                raise ValueError('Report path already exists or conflicts with input/output')
        expected = None
        if args.expected:
            if args.input.is_dir() and args.expected.resolve().is_relative_to(args.input.resolve()):
                raise ValueError('Expected image list must be outside the input package')
            expected = load_json(args.expected.read_bytes())
            image_records(expected)
        if args.runtime:
            if args.output is None:
                raise ValueError('--output is required with --runtime')
            report = decode(args.input, args.output, args.runtime, expected, args.budget_bpp, args.timeout)
        elif check:
            manifest, _, summary = read_package(args.input, expected, args.budget_bpp)
            report = dict(package=summary, decode_executed=False)
            if args.output:
                report['output'] = check_images(args.output, image_records(expected))
        else:
            raise ValueError('--runtime and --output are required to decode')
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open('x', encoding='utf-8') as handle:
                handle.write(text + '\n')
        print(text)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'ERROR: {exc}\n')
