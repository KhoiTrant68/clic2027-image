"""Check transmitted package against an independent expected image list."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from clic27.submission import main

if __name__ == '__main__':
    main(check=True)
