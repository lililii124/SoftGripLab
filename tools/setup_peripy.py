"""Fetch pinned MIT-licensed PeriPy sources; kernels are loaded without modification."""
import argparse
from pathlib import Path
import subprocess

COMMIT = "04cae072e22568e29b9ab8f84d821f9576ecb728"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    target = args.destination.resolve()
    if not target.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/alan-turing-institute/PeriPy.git",
                str(target),
            ],
            check=True,
        )
    if not (target / ".git").is_dir():
        parser.error("destination exists but is not a git checkout")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=target, text=True
    )
    if dirty.strip():
        parser.error("destination has local changes; choose another directory")
    subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=target, check=True)
    print(
        f"PeriPy source: {target}\nRevision: {COMMIT}\nPass this path as banana-cutting --backend."
    )


if __name__ == "__main__":
    main()
