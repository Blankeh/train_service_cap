import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "src.main"] + sys.argv[1:]
    result = subprocess.run(cmd, cwd=ROOT)
    sys.exit(result.returncode)
