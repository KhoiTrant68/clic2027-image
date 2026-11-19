"""Seed an independent round with evenly allocated pairs and images."""
import argparse
from collections import Counter
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import re
import secrets
import shutil


def insert(table, values):
    fields = ', '.join("'" + str(value).replace("'", "''") + "'" for value in values)
    return f'INSERT INTO {table} VALUES ({fields});\n'


def main():
    parser = argparse.ArgumentParser()
    for name in ['assets', 'out', 'public']:
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--round', default='clic30-v2', help='Immutable URL slug for this round')
    parser.add_argument('--model', action='append', help='Include this model ID (repeat for each model)')
    parser.add_argument('--rate', type=float, action='append', help='Include this target bpp (repeat for each rate)')
    args = parser.parse_args()
    manifest = json.loads((args.assets / 'manifest.json').read_text(encoding='utf-8'))
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,39}', args.round):
        parser.error('Round ID must use lowercase letters, digits and hyphens')
    rates = sorted(set(args.rate or [0.075, 0.15, 0.3]))
    models = args.model or ['original', 'perceptual', 'single', 'mixed']
    if len(set(models)) != len(models) or len(models) < 2 or not 1 <= len(rates) <= 6:
        parser.error('Supply at least two distinct model IDs and up to six rates')
    if args.round == 'clic30-v2' and (args.model or args.rate):
        parser.error('Use a new round ID for a different model list or bitrate')
    by_rate = {rate: [row for row in manifest if row['target_bpp'] == rate] for rate in rates}
    source_sets = [{row['source'] for row in records} for records in by_rate.values()]
    if any(len(rows) != 30 or len(sources) != 30 or sources != source_sets[0]
           for rows, sources in zip(by_rate.values(), source_sets)):
        raise ValueError('Each rate must have the same 30 distinct source images')
    selected = [row for rows in by_rate.values() for row in rows]
    opaque = {asset for row in selected for asset in [row['reference'], *(row['models'][model] for model in models)]}
    (args.public / 'asset').mkdir(parents=True, exist_ok=True)
    for key in opaque:
        if not re.fullmatch(r'[a-f0-9]{48}', key):
            raise ValueError('Bad opaque asset ID')
        dest = args.public / 'asset' / f'{key}.png'
        if not dest.exists():
            source = args.assets / 'asset' / dest.name
            if source.stat().st_size > 25*1024**2:
                raise ValueError('PNG is too large for static assets')
            shutil.copyfile(source, dest)
    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261111)
    pair_count, image_count, both_count = Counter(), Counter(), Counter()
    pairs = list(combinations(models, 2))
    if len(pairs) > 100:
        parser.error('20 schedules × 5 questions per rate cannot cover more than 100 pairs')
    offsets = {(rate, pair): rng.randrange(2) for rate in rates for pair in pairs}
    with (args.out / 'seed.sql').open('w', encoding='utf-8') as sql, \
            (args.out / 'invitations.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        sql.write('''CREATE TABLE IF NOT EXISTS trial_context (
          trial TEXT PRIMARY KEY REFERENCES trials(id), study TEXT NOT NULL,
          dataset TEXT NOT NULL, target_bpp REAL NOT NULL,
          image_bytes INTEGER NOT NULL, pixels INTEGER NOT NULL, stream_sha256 TEXT NOT NULL
        );\n''')
        invites = csv.writer(handle)
        invites.writerow(['participant', 'invitation'])
        for index in range(20):
            prefix = 'V2' if args.round == 'clic30-v2' else f'ROUND-{args.round}'
            participant = f'{prefix}-P{index+1:02}'
            code = secrets.token_hex(16)
            invites.writerow([participant, code])
            sql.write(insert('invitations', [hashlib.sha256(code.encode()).hexdigest(), participant]))
            assigned, used = [], set()
            for rate_index, rate in enumerate(rates):
                for slot in range(5):
                    pair = pairs[(index*5 + slot + rate_index) % len(pairs)]
                    pool = [row for row in by_rate[rate] if row['source'] not in used]
                    rng.shuffle(pool)
                    row = min(pool, key=lambda entry: (image_count[rate, entry['source']], both_count[rate, pair, entry['source']]))
                    name = row['source']
                    used.add(name)
                    a, b = pair[::-1] if (pair_count[rate, pair] + offsets[rate, pair]) % 2 else pair
                    pair_count[rate, pair] += 1
                    image_count[rate, name] += 1
                    both_count[rate, pair, name] += 1
                    assigned.append((row, a, b))
            rng.shuffle(assigned)
            for position, (row, a, b) in enumerate(assigned):
                trial = secrets.token_hex(16)
                sql.write(insert('trials', [trial, participant, position, row['source'], row['reference'],
                                           row['models'][a], row['models'][b], a, b]))
                sql.write(insert('trial_context', [trial, args.round, row['dataset'], row['target_bpp'],
                                                  row['bytes'], row['pixels'], row['stream_sha256']]))
    print(f'{sum(pair_count.values())} trials; {len(pair_count)} rate/pair groups; {len(opaque)} opaque assets')


if __name__ == '__main__':
    main()
