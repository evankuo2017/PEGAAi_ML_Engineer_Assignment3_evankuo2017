"""Entry point for the Smart Factory Alert Agent."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from agent import main

if __name__ == "__main__":
    main()
