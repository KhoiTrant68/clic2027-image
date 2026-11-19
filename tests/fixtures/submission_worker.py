"""Test double only: .bin files contain PNGs, never real codec bitstreams."""
import argparse
import json
import io
from pathlib import Path
import sys
import time

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from clic27.codec import msillm_stream
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/bakeoff'))
import pulse_codec

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--metadata', type=Path, required=True)
parser.add_argument('--mode', default='ok')
args = parser.parse_args()
if args.mode == 'fail':
    print('intentional worker failure', flush=True)
    sys.exit(7)
if args.mode == 'timeout':
    time.sleep(10)
args.output.mkdir()
records = json.loads(args.metadata.read_text())['images']
for row in records:
    if args.mode == 'missing':
        continue
    if args.mode == 'invalid-png':
        (args.output / (row['name'] + '.png')).write_bytes(b'not a PNG')
        continue
    data = (args.input / (row['name'] + '.bin')).read_bytes()
    if row['codec'] == 'msillm':
        data = msillm_stream.unpack(data)[3]
    elif row['codec'] == 'pulse':
        data = pulse_codec.unpack(data)
    with Image.open(io.BytesIO(data)) as image:
        if args.mode == 'wrong-size':
            image = image.resize((1, 1))
        if args.mode == 'wrong-pixels':
            image = Image.new('RGB', image.size, 'red')
        if args.mode == 'wrong-mode':
            image = image.convert('L')
        image.save(args.output / (row['name'] + '.png'))
if args.mode == 'extra':
    Image.new('RGB', (1, 1)).save(args.output / 'unexpected.png')
