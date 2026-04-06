from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    test_file = Path(__file__).with_name("test_chapter34_interfaces.py")
    result = subprocess.run([sys.executable, "-m", "unittest", str(test_file)], check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
