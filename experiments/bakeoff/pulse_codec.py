"""PULSE stream framing shared by bake-off and the standalone decoder."""
import hashlib
from pathlib import Path

# Four transmitted bytes: codec id, version, reserved flags. PULSE carries QP,
# dimensions and bundle fingerprint inside its own checked envelope.
HEADER = b"PL\x01\x00"


def pack(payload):
    if not payload:
        raise ValueError("Empty PULSE payload")
    return HEADER + payload


def unpack(data):
    if len(data) <= len(HEADER) or data[:4] != HEADER:
        raise ValueError("Unknown codec/version/flags or truncated stream")
    return data[4:]


def bundle_hashes(directory):
    result = {}
    for name in ("config.json", "model.pt", "entropy_control_int.pt"):
        h = hashlib.sha256()
        with (Path(directory) / name).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        result[name] = h.hexdigest()
    return result
