"""Convert the organizer's Wrangler D1 JSON export to CSV and pair summaries."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


CONTEXT = ['study', 'dataset', 'target_bpp', 'image_bytes', 'pixels', 'stream_sha256']


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        pair = tuple(sorted((row['codec_a'], row['codec_b'])))
        groups[row.get('study', 'clic30-v1'), row.get('dataset', 'clic30-full'),
               row.get('target_bpp', 0.15), pair].append(row)
    summary = []
    for (study, dataset, rate, pair), votes in sorted(groups.items()):
        wins = Counter({codec: 0 for codec in pair})
        others = Counter()
        for vote in votes:
            choice = vote['choice']
            if choice in {'A', 'B'}:
                wins[vote[f'codec_{choice.lower()}']] += 1
            else:
                others[choice] += 1
        summary.append({'study': study, 'dataset': dataset, 'target_bpp': rate, 'pair': pair,
                        'votes': len(votes), 'participants_with_votes': len({v['participant'] for v in votes}),
                        'wins': wins, 'ties': others['tie'], 'skipped': others['skip']})
    return {'votes': len(rows), 'participants_with_votes': len({row['participant'] for row in rows}),
            'by_study_rate_pair': summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--versioned', action='store_true', help='Include round columns even if there are no votes')
    args = parser.parse_args()
    rows = json.loads(args.source.read_text(encoding='utf-8-sig'))[0]['results']
    args.out.mkdir(parents=True, exist_ok=False)
    fields = ['participant', 'source', 'codec_a', 'codec_b', 'choice', 'seconds', 'timestamp_ms']
    if args.versioned or (rows and 'study' in rows[0]):
        fields = CONTEXT + fields
    with (args.out / 'votes.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / 'summary.json').write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
