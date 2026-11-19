"""Compatibility import for research runners; framing is shared with the dispatcher."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from clic27.codec.msillm_stream import HEADER, pack, unpack  # noqa: E402,F401
