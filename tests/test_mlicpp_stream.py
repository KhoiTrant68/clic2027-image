import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('mlicpp_stream', Path(__file__).resolve().parents[1] / 'experiments/mlicpp/stream.py')
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)


class MlicStreamTest(unittest.TestCase):
    def test_roundtrip_and_transmitted_overhead(self):
        args = (1, True, (1360, 2048), (22, 32), [[b'latent'], [b'hyper']])
        blob = stream.pack(*args)
        self.assertEqual(stream.unpack(blob), args)
        self.assertEqual(len(blob), 31 + len(b'latenthyper'))

    def test_corruption_and_inconsistent_shapes(self):
        blob = stream.pack(2, True, (65, 64), (2, 1), [[b'a'], [b'b']])
        for bad in (b'', blob[:12], blob[:-1], blob+b'x', b'XXXX'+blob[4:], blob[:4]+b'\x02'+blob[5:]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                stream.unpack(bad)
        with self.assertRaises(ValueError):
            stream.pack(1, True, (65, 64), (1, 1), [[b'a'], [b'b']])
        with self.assertRaises(ValueError):
            stream.pack(1, True, (64, 64), (1, 1), [[b'a'], [b'']])


if __name__ == '__main__':
    unittest.main()
