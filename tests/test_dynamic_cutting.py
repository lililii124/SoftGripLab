"""Finite moving-blade cuts must preserve material on the same blade side."""
import numpy as np
import pytest
from softgriplab.backends.peridynamic_cutting import (
    CUT_KERNEL,
    blade_mesh,
    material_points,
)


def test_initial_body_is_supported_and_connected():
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    spacing = 0.0025
    x = material_points(spacing)
    pairs = cKDTree(x).query_pairs(2.51 * spacing, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), pairs.T), shape=(len(x), len(x)))
    assert connected_components(graph, directed=False, return_labels=False) == 1
    assert x[:, 2].min() == pytest.approx(0.45 * spacing)


def test_blade_only_cuts_crossing_bonds_inside_its_current_extent():
    cl = pytest.importorskip("pyopencl")
    try:
        devices = [
            d
            for p in cl.get_platforms()
            for d in p.get_devices()
            if d.double_fp_config and "FPGA" not in d.name
        ]
    except cl.LogicError:
        pytest.skip("No OpenCL platform")
    if not devices:
        pytest.skip("No double precision OpenCL device")
    ctx = cl.Context([devices[0]])
    q = cl.CommandQueue(ctx)

    def buf(a):
        return cl.Buffer(
            ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(a),
        )

    kernel = cl.Program(ctx, CUT_KERNEL).build().knife_cut
    # Across blade, same side within wedge, below tip, beyond blade end.
    x = np.array(
        [
            [-0.003, 0, 0.01],
            [0.003, 0, 0.01],
            [-0.0002, 0, 0.02],
            [-0.0003, 0, 0.02],
            [-0.003, 0, -0.01],
            [0.003, 0, -0.01],
            [-0.003, 0.08, 0.01],
            [0.003, 0.08, 0.01],
        ]
    )
    initial = np.array([[1], [0], [3], [2], [5], [4], [7], [6]], np.int32)
    planes = blade_mesh()[2]
    for center, broken in [
        (np.zeros(3), [0, 1]),
        (np.array([0, 0.08, 0]), [6, 7]),
        (np.array([0, 0, 0.2]), []),
    ]:
        nl = initial.copy()
        nd = buf(nl)
        kernel(
            q,
            nl.shape,
            None,
            buf(x),
            buf(np.zeros_like(x)),
            nd,
            buf(planes),
            np.int32(1),
            np.int32(len(planes)),
            *map(np.float64, [*center, 1, 0, 0]),
        )
        cl.enqueue_copy(q, nl, nd).wait()
        expected = initial.copy()
        expected[broken] = -1
        np.testing.assert_array_equal(nl, expected)
