"""Local mix entrypoint; package src/clic27 and actual workers alongside deployment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from clic27.submission import main

if __name__ == '__main__':
    main()
