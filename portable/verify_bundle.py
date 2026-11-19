import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
records=json.loads((root/'MANIFEST.sha256.json').read_text())
for name,expected in records.items():
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Unsafe manifest path')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Hash mismatch: '+name)
print(f'OK: {len(records)} files verified')
