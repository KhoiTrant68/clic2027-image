"""The mode header is transmitted and rejected on corruption/version mismatch."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "pulse_codec", Path(__file__).resolve().parents[1] / "experiments/bakeoff/pulse_codec.py")
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


class FramingTest(unittest.TestCase):
    def test_roundtrip_and_byte_accounting(self):
        payload = bytes(range(256))
        stream = codec.pack(payload)
        self.assertEqual(len(stream), len(payload) + 4)
        self.assertEqual(codec.unpack(stream), payload)

    def test_reject_truncated_unknown_version_and_flags(self):
        for bad in (b"", b"PL", codec.HEADER, b"PL\x02\x00x", b"PL\x01\x01x", b"XX\x01\x00x"):
            with self.subTest(data=bad), self.assertRaises(ValueError):
                codec.unpack(bad)
        with self.assertRaises(ValueError):
            codec.pack(b"")


if __name__ == "__main__":
    unittest.main()
