"""Build the optional NVIDIA research backend outside the SoftGripLab tree.

Python 3.11, CPU PyTorch, Ninja and a C++ compiler are required. This script
does not redistribute DiSECt or change its license. See docs/CUTTING.md.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT = "ff90165070335bb8033000373d2d20c6aac13edd"
URL = "https://github.com/NVlabs/DiSECt.git"


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def find_vcvars():
    supplied = os.environ.get("SOFTGRIPLAB_VCVARS")
    if supplied and Path(supplied).is_file():
        return supplied
    for base in (
        os.environ.get("ProgramFiles", "C:/Program Files"),
        os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"),
    ):
        matches = sorted(
            Path(base).glob(
                "Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat"
            ),
            reverse=True,
        )
        if matches:
            return str(matches[0])
    raise RuntimeError(
        "Install Visual Studio C++ Build Tools, or set SOFTGRIPLAB_VCVARS to vcvars64.bat"
    )


def prepare(root):
    if root.exists():
        raise FileExistsError(
            f"Destination already exists: {root}. Choose a new --destination; existing checkouts are never overwritten."
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", URL, str(root)],
        check=True,
    )
    subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=root, check=True)
    subprocess.run(
        ["git", "sparse-checkout", "set", "cutting", "dflex", "meshing"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=root, check=True)
    if git(root, "rev-parse", "HEAD") != COMMIT:
        raise RuntimeError("Unexpected backend revision")

    changed = {}

    def patch(name, replacements):
        path = root / name
        original = path.read_text(encoding="utf-8")
        result = original
        for old, new in replacements:
            if old not in result:
                raise RuntimeError(
                    f"Compatibility patch no longer matches {name}: {old!r}"
                )
            result = result.replace(old, new)
        path.write_text(result, encoding="utf-8")
        changed[name] = {
            "upstream_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "patched_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    adjoint = [
        ("node.slice.value", "node.slice"),
        ('["-Z", "-O2", "-DNDEBUG"]', '["-O2", "-DNDEBUG"]'),
        ('ld_flags = ["-DNDEBUG"]', "ld_flags = []"),
    ]
    if os.name == "nt":
        original = (root / "dflex/adjoint.py").read_text(encoding="utf-8")
        old = next(line for line in original.splitlines() if "vcvars_path = " in line)
        adjoint.append((old, "        vcvars_path = " + repr(find_vcvars())))
    patch("dflex/adjoint.py", adjoint)
    patch(
        "dflex/__init__.py",
        [
            (
                "from dflex.render import *",
                "# Headless adapter: USD renderer is not imported.",
            )
        ],
    )
    patch(
        "cutting/__init__.py",
        [
            (
                "from .visualizer import Visualizer",
                "# Headless adapter: interactive GUI is not imported.",
            )
        ],
    )
    patch(
        "cutting/settings.py",
        [
            (
                "from collections import MutableMapping, namedtuple, defaultdict",
                "from collections import namedtuple, defaultdict\nfrom collections.abc import MutableMapping",
            )
        ],
    )
    patch(
        "cutting/urdf_loader.py",
        [
            ("import urdfpy", "# urdfpy is loaded only when its loader is called."),
            (
                "    robot = urdfpy.URDF.load(filename)",
                "    import urdfpy\n    robot = urdfpy.URDF.load(filename)",
            ),
        ],
    )
    # Also make the optional Python mesher's collection imports explicit.
    patch(
        "dflex/model.py",
        [
            (
                "from copy import copy",
                "from copy import copy\nfrom collections import defaultdict, OrderedDict",
            )
        ],
    )
    manifest = {
        "commit": COMMIT,
        "license": "NVIDIA Source Code License, research/evaluation only; see LICENSE.md",
        "compatibility_patches": changed,
        "python": sys.version,
        "numerical_kernels_modified": False,
    }
    (root / "softgriplab_setup.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def build(root):
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = [str(root), str(root / "meshing/build")]
    import torch

    if torch.version.cuda is not None:
        raise RuntimeError("Use a CPU-only PyTorch environment for this tested adapter")
    import dflex
    from torch.utils.cpp_extension import load

    dflex.adjoint.set_build_env()
    build_dir = root / "meshing/build"
    build_dir.mkdir(parents=True, exist_ok=True)
    load(
        name="meshcutter",
        sources=[str(root / "meshing/meshing.cpp")],
        extra_cflags=["/std:c++17", "/O2"]
        if os.name == "nt"
        else ["-std=c++17", "-O2"],
        build_directory=str(build_dir),
        verbose=True,
    )
    from cutting.cutting_sim import CuttingSim  # noqa: F401

    manifest_path = root / "softgriplab_setup.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(torch=torch.__version__, build_complete=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Backend ready: {root}\nRun: python -m softgriplab banana-cutting --backend {root}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path.home() / ".cache/softgriplab/disect-cpu",
    )
    parser.add_argument(
        "--resume-build",
        action="store_true",
        help="Retry compilation in a checkout prepared by this script",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error(
            "This optional legacy backend is tested with Python 3.11. Use a separate environment."
        )
    root = args.destination.resolve()
    print(
        "DiSECt retains its NVIDIA research/evaluation license: https://github.com/NVlabs/DiSECt/blob/main/LICENSE.md",
        flush=True,
    )
    if args.resume_build:
        manifest = json.loads(
            (root / "softgriplab_setup.json").read_text(encoding="utf-8")
        )
        if manifest["commit"] != COMMIT or git(root, "rev-parse", "HEAD") != COMMIT:
            raise RuntimeError("Backend revision does not match")
        for name, hashes in manifest["compatibility_patches"].items():
            if (
                hashlib.sha256((root / name).read_bytes()).hexdigest()
                != hashes["patched_sha256"]
            ):
                raise RuntimeError(f"Patched file was changed: {name}")
    else:
        prepare(root)
    build(root)


if __name__ == "__main__":
    main()
