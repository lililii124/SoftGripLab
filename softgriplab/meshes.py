"""Structured tetrahedral specimens and their boundary mesh (SI units)."""
from itertools import permutations
from pathlib import Path
import numpy as np


def tetrahedral_block(size=(0.05, 0.05, 0.04), cells=(6, 6, 5)):
    axes = [
        np.linspace(-length / 2, length / 2, n + 1) for length, n in zip(size, cells)
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    ids = np.arange(len(points)).reshape(tuple(n + 1 for n in cells))
    tets = []
    for base in np.ndindex(*cells):
        for order in permutations(range(3)):
            vertex = np.array(base)
            tet = [ids[tuple(vertex)]]
            for direction in order:
                vertex[direction] += 1
                tet.append(ids[tuple(vertex)])
            if np.linalg.det((points[tet[1:]] - points[tet[0]]).T) < 0:
                tet[1], tet[2] = tet[2], tet[1]
            tets.append(tet)
    tets = np.asarray(tets, dtype=np.int32)
    boundary = {}
    for a, b, c, d in tets:
        for face in ((a, c, b), (a, b, d), (a, d, c), (b, c, d)):
            key = tuple(sorted(face))
            if key in boundary:
                boundary[key] = None
            else:
                boundary[key] = face
    faces = np.array(
        [face for face in boundary.values() if face is not None], dtype=np.int32
    )
    return points, tets, faces


def write_mesh(directory, points, tets, faces):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    obj, vtk = directory / "specimen.obj", directory / "specimen.vtk"
    obj.write_text(
        "".join("v %.12g %.12g %.12g\n" % tuple(v) for v in points)
        + "".join("f %d %d %d\n" % tuple(f + 1) for f in faces),
        encoding="ascii",
    )
    vtk.write_text(
        "# vtk DataFile Version 2.0\nSoftGripLab tetrahedral specimen\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        + f"POINTS {len(points)} double\n"
        + "".join("%.12g %.12g %.12g\n" % tuple(v) for v in points)
        + f"CELLS {len(tets)} {5 * len(tets)}\n"
        + "".join("4 %d %d %d %d\n" % tuple(t) for t in tets)
        + f"CELL_TYPES {len(tets)}\n"
        + "10\n" * len(tets),
        encoding="ascii",
    )
    return obj, vtk
