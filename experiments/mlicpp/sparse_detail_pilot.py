"""Prototype sparse luminance detail packets with exact ZIP byte accounting."""
import argparse
import io
import json
from pathlib import Path
import struct
import sys
import zipfile

import numpy as np
from PIL import Image

from budget_blind import package
from probe import ROOT, sha, write_json


BLOCK = 16
STEP = 8
RECORD = struct.Struct('<BIBb')
MAGIC = b'SDT1'
MEMBER = 'detail.bin'
FREQUENCIES = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (0, 3), (3, 0), (2, 2)]


def codebook():
    axis = np.arange(BLOCK) + .5
    atoms = []
    for y, x in FREQUENCIES:
        atom = np.outer(np.cos(np.pi * y * axis / BLOCK), np.cos(np.pi * x * axis / BLOCK))
        atoms.append(np.rint(atom / np.linalg.norm(atom) * 1024).astype(np.float32) / 1024)
    return np.stack(atoms)


ATOMS = codebook()


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def candidates(reference, base, image_id):
    # Partial tiles are left intact, so the decoder needs no boundary metadata.
    rows, cols = (n // BLOCK for n in reference.shape[:2])
    h, w = rows * BLOCK, cols * BLOCK
    luma = np.array([.299, .587, .114], dtype=np.float32)
    target = reference[:h, :w] @ luma
    residual = target - base[:h, :w] @ luma
    edge = np.abs(np.diff(target, axis=0, prepend=target[:1]))
    edge += np.abs(np.diff(target, axis=1, prepend=target[:, :1]))

    def tiles(x):
        return x.reshape(rows, BLOCK, cols, BLOCK).transpose(0, 2, 1, 3).reshape(-1, BLOCK**2)

    weights = 1 + np.minimum(tiles(edge) / 32, 2)
    atoms = ATOMS.reshape(len(ATOMS), -1)
    correlation = (weights * tiles(residual)) @ atoms.T
    energy = weights @ (atoms**2).T
    amplitudes = np.clip(np.rint(correlation / energy / STEP), -127, 127).astype(np.int16)
    shift = amplitudes.astype(np.float32) * STEP
    gains = 2 * correlation * shift - energy * shift**2
    atom_ids = gains.argmax(axis=1)
    return [(float(gains[i, a]), image_id, i, int(a), int(amplitudes[i, a]))
            for i, a in enumerate(atom_ids) if gains[i, a] > 0]


def apply_packet(base, packet, image_id):
    if not packet.startswith(MAGIC) or (len(packet) - len(MAGIC)) % RECORD.size:
        raise ValueError('Invalid detail packet')
    rows, cols = (n // BLOCK for n in base.shape[:2])
    output = base.copy()
    used = set()
    for image, tile, atom, amplitude in RECORD.iter_unpack(packet[len(MAGIC):]):
        if image != image_id:
            continue
        if tile >= rows * cols or atom >= len(ATOMS) or tile in used or not amplitude:
            raise ValueError('Invalid or duplicate detail token')
        used.add(tile)
        y, x = divmod(tile, cols)
        block = output[y*BLOCK:(y+1)*BLOCK, x*BLOCK:(x+1)*BLOCK]
        change = np.rint(ATOMS[atom] * (STEP * amplitude)).astype(np.int16)
        block[:] = np.clip(block.astype(np.int16) + change[:, :, None], 0, 255)
    return output


def with_packet(base_zip, packet):
    buffer = io.BytesIO(base_zip)
    with zipfile.ZipFile(buffer, 'a', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(MEMBER, packet)
    return buffer.getvalue()


def allocate(points, cap):
    selected = [min(choices, key=lambda r: r['bytes']) for choices in points]
    size = len(package(selected, 1))
    if size > cap:
        raise ValueError('Even the smallest base package exceeds the cap')
    while True:
        best = None
        for i, old in enumerate(selected):
            for new in points[i]:
                cost = new['bytes'] - old['bytes'] + len(str(new['factor'])) - len(str(old['factor']))
                gain = old['lpips'] - new['lpips']
                if gain > 0 and size + cost <= cap:
                    score = gain / max(cost, 1)
                    if best is None or score > best[0]:
                        best = (score, i, new, cost)
        if best is None:
            break
        _, index, record, cost = best
        selected[index] = record
        size += cost
    blob = package(selected, 1)
    if len(blob) != size or size > cap:
        raise ValueError('Exact ZIP size accounting failed')
    return selected, blob


def run(args):
    grid = args.source / 'grid'
    names = {}
    for factor in (.5, 1., 2., 4.):
        folder = grid / f'perceptual-factor{factor:g}'
        for row in json.loads((folder / 'metrics.json').read_text()):
            names.setdefault(row['name'], []).append(dict(
                row, factor=factor, stream_path=folder/'streams'/row['stream'],
                recon_path=folder/'recon'/row['name']))
    points = [names[name] for name in sorted(names)]
    if len(points) != 30 or any(len(group) != 4 for group in points):
        raise ValueError('Expected all four operating points for each CLIC30 image')
    cap = (args.source/'budgets'/'bpp-0.075'/'perceptual.zip').stat().st_size
    full, full_zip = allocate(points, cap)
    cap = len(full_zip)
    base, base_zip = allocate(points, int(cap * .96))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'images').mkdir()
    reference_dir = grid/'development32-full'/'images'
    hashes = {r['name']: r['sha256'] for r in json.loads((reference_dir.parent/'manifest.json').read_text())['images']}
    pool = []
    for i, row in enumerate(base):
        ref = reference_dir/row['name']
        if sha(ref) != hashes[row['name']]:
            raise ValueError(f'Reference hash mismatch: {ref}')
        pool.extend(candidates(read_rgb(ref), read_rgb(row['recon_path']), i))
        write_json(args.out/'status.json', dict(stage='select', images=i+1, total=len(base)))
    overhead = len(with_packet(base_zip, MAGIC))
    capacity = (cap - overhead) // RECORD.size
    if capacity <= 0 or not pool:
        raise ValueError('No room for detail tokens')
    selected = sorted(sorted(pool, reverse=True)[:capacity], key=lambda r: r[1:3])
    packet = MAGIC + b''.join(RECORD.pack(*entry[1:]) for entry in selected)
    augmented = with_packet(base_zip, packet)
    if len(augmented) > cap:
        raise ValueError('Detail package exceeds baseline cap')
    (args.out/'base-full.zip').write_bytes(full_zip)
    (args.out/'base-reduced.zip').write_bytes(base_zip)
    (args.out/'base-plus-detail.zip').write_bytes(augmented)
    with zipfile.ZipFile(args.out/'base-plus-detail.zip') as archive:
        assert archive.testzip() is None
        packet = archive.read(MEMBER)

    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT/'src'))
    from clic27.eval.metrics import Perceptual, psnr

    metric = Perceptual('cpu')
    measured = []
    for i, (low, high) in enumerate(zip(base, full)):
        target = read_rgb(reference_dir/low['name'])
        output = apply_packet(read_rgb(low['recon_path']), packet, i)
        Image.fromarray(output).save(args.out/'images'/low['name'])
        measured.append(dict(name=low['name'], base_factor=low['factor'], full_factor=high['factor'],
            detail=dict(lpips=metric.lpips(target, output), dists=metric.dists(target, output), psnr=psnr(target, output)),
            reduced={m: low[m] for m in ('lpips', 'dists', 'psnr')},
            full={m: high[m] for m in ('lpips', 'dists', 'psnr')}))
        write_json(args.out/'status.json', dict(stage='evaluate', images=i+1, total=len(base)))
        write_json(args.out/'per_image.json', measured)
    pixels = sum(row['pixels'] for row in base)
    summary = dict(images=len(base), cap_bytes=cap, full_bytes=len(full_zip), reduced_bytes=len(base_zip),
        detail_bytes=len(augmented), side_bytes=len(augmented)-len(base_zip), records=len(selected),
        full_bpp=8*len(full_zip)/pixels, detail_bpp=8*len(augmented)/pixels,
        metrics={arm: {m: float(np.mean([r[arm][m] for r in measured])) for m in ('lpips','dists','psnr')}
                 for arm in ('full','reduced','detail')})
    write_json(args.out/'summary.json', summary)
    write_json(args.out/'status.json', dict(stage='complete', images=len(base)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
