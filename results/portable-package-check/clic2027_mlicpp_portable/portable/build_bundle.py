"""Build a source-only archive with an explicit allowlist and integrity manifest."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

root=Path(__file__).resolve().parents[1]
files=set()
for directory in ['portable','experiments/mlicpp','src/clic27']:
    for path in (root/directory).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py','.sh','.json','.md','.html'}:
            if path.name not in {'local.json','requirements.installed.txt'}:files.add(path)
for name in ['pyproject.toml','experiments/new_codecs/half_vram.py','tools/blind_score.py','tools/BLIND_SCORING.md',
    'tests/test_mlicpp_stream.py','tests/test_blind_score.py',
    'reports/MLICpp_CPU_development_2026-11-07.md','reports/MLICpp_end_to_end_2026-11-07.md',
    'reports/MLICpp_full_synthesis_pilot_2026-11-05.md','reports/CLIC2027_MLICpp_full30_2026-11-02.md',
    'plans/CLIC2027_competitive_path_2026-11-07.md','plans/MLICpp_end_to_end_2026-11-07.md',
    'results/2026-11-07_mlicpp_end_to_end/cpu_dev8_comparison.json','results/2026-11-07_mlicpp_end_to_end/smoke.json']:
    files.add(root/name)
payload={}
for path in sorted(files):
    raw=path.read_bytes()
    # Linux scripts and Python/text assets are normalized to LF.
    raw=raw.replace(b'\r\n',b'\n')
    if re.search(rb'nxs_[A-Za-z0-9_-]{20,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',raw):
        raise ValueError('Credential-like content in '+str(path.relative_to(root)))
    payload[path.relative_to(root).as_posix()]=raw
payload['README.md']=b'# CLIC 2027 / MLIC++ portable bundle\n\nStart with portable/HUONG_DAN.md (Vietnamese).\nCode only: datasets and model weights are transferred or downloaded separately.\n'
payload['MANIFEST.sha256.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in payload.items()},indent=2).encode()
dest=root/'dist';dest.mkdir(exist_ok=True)
archive=dest/'clic2027_mlicpp_portable_2026-11-08.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in payload.items():
        info=zipfile.ZipInfo('clic2027_mlicpp_portable/'+name)
        info.compress_type=zipfile.ZIP_DEFLATED
        info.external_attr=(0o100755 if name.endswith('.sh') else 0o100644)<<16
        z.writestr(info,data)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,data in payload.items():assert z.read('clic2027_mlicpp_portable/'+name)==data
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
print(json.dumps(dict(path=str(archive),files=len(payload),bytes=archive.stat().st_size,sha256=digest)))
