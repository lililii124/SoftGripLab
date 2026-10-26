"""Moving-knife cutting with the unmodified PeriPy force/damage/integration kernels.

The initial material has no cut interfaces. Bonds intersected by the measured
finite blade are severed during integration. This is a geometric cutting rule,
not a prediction of fracture toughness. Bulk motion is solved by PeriPy.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

PERIPY_COMMIT = "04cae072e22568e29b9ab8f84d821f9576ecb728"
KERNEL_SHA256 = {
    "peripy/cl/peridynamics.cl": "b08f8f74a502a66bff35077d7eab0e243dc64cbe49b96b945f61b7e01f657914",
    "peripy/cl/euler_cromer.cl": "73bed7b723dec79375667506f67c9106bc47fb174310070630b172956093aca8",
}
OFFSET = np.array([0.5, 0.0, 0.1])

# Contact adapter only. PeriPy supplies the material law and time integrator.
CUT_KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void knife_cut(__global const double *r, __global const double *u,
 __global int *nl, __global const double *planes, int width, int nf,
 double cx, double cy, double cz, double nx, double ny, double nz) {
 int i=get_global_id(0), k=get_global_id(1), j=nl[i*width+k];
 if(j<0) return;
 double3 a=vload3(i,r)+vload3(i,u)-(double3)(cx,cy,cz);
 double3 d=vload3(j,r)+vload3(j,u)-(double3)(cx,cy,cz)-a;
  // A cut separates the two blade sides. Removing every bond inside the
  // wedge also erodes intact material on each side and creates loose grains.
  double3 normal=(double3)(nx,ny,nz);
  double sa=dot(normal,a), sb=dot(normal,a+d);
  if((sa>=0.0)==(sb>=0.0)) return;
  double crossing=-sa/(sb-sa);
  double3 intersection=a+crossing*d;
  for(int f=0; f<nf; ++f) {
    double4 p=vload4(f,planes);
    if(dot(p.xyz,intersection)+p.w>1e-12) return;
  }
  nl[i*width+k]=-1;
  return;
}
"""


def material_points(spacing):
    axes = [
        np.arange(a + spacing / 2, b, spacing)
        for a, b in [(-0.08, 0.08), (-0.03, 0.05), (0.001, 0.032)]
    ]
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    s = xyz[:, 0] / 0.08
    r = 0.014 * (1 - 0.6 * s**4)
    curve = 0.035 * (1 - s * s) - 0.0175
    xyz = xyz[(xyz[:, 1] - curve) ** 2 + (xyz[:, 2] - 0.016) ** 2 < r * r]
    # Start in contact with the support, then settle under gravity off camera.
    xyz[:, 2] += 0.45 * spacing - xyz[:, 2].min()
    return np.ascontiguousarray(xyz)


def blade_mesh():
    vertices = np.array(
        [
            [x, y, z]
            for y in (-0.055, 0.055)
            for x, z in [(-0.0001, 0), (0.0001, 0), (0.0015, 0.04), (-0.0015, 0.04)]
        ]
    )
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    tri = vertices[faces]
    inward = (
        np.einsum(
            "ij,ij->i",
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            hull.equations[:, :3],
        )
        < 0
    )
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return vertices, faces, np.unique(np.round(hull.equations, 12), axis=0)


def tool_target(time_s, cut_positions=(0.045, 0.0, -0.045)):
    times, positions = [0.0], [[cut_positions[0], 0.0, 0.04]]
    for i, x in enumerate(cut_positions):
        t = i * 1.4
        times.extend([t + 0.15, t + 0.80, t + 0.95, t + 1.2])
        positions.extend([[x, 0, 0.04], [x, 0, 0], [x, 0, 0], [x, 0, 0.04]])
        if i < len(cut_positions) - 1:
            times.append(t + 1.4)
            positions.append([cut_positions[i + 1], 0, 0.04])
    times.append(len(cut_positions) * 1.4)
    positions.append(positions[-1])
    i = int(
        np.clip(np.searchsorted(times, time_s, side="right") - 1, 0, len(times) - 2)
    )
    u = np.clip((time_s - times[i]) / (times[i + 1] - times[i]), 0, 1)
    s = u * u * (3 - 2 * u)
    return np.asarray(positions[i]) + s * (np.asarray(positions[i + 1]) - positions[i])


class PeriPyMaterial:
    def __init__(self, source, spacing, dt, critical_stretch, knife_cutting):
        import pyopencl as cl

        self.cl = cl
        self.reference = material_points(spacing)
        self.spacing, self.dt = spacing, dt
        self.rho, self.young = 1000.0, 50000.0
        self.n = len(self.reference)
        self.vol = np.full(self.n, spacing**3)
        self.mass = self.rho * self.vol
        self.horizon = 2.51 * spacing
        family = cKDTree(self.reference).query_ball_point(self.reference, self.horizon)
        family = [sorted(set(row) - {i}) for i, row in enumerate(family)]
        self.width = 1 << (max(map(len, family)) - 1).bit_length()
        self.nl = np.full((self.n, self.width), -1, np.int32)
        for i, row in enumerate(family):
            self.nl[i, : len(row)] = row
        self.initial = self.nl.copy()
        self.initial_count = int((self.nl >= 0).sum())
        # Column lookup is used only to distinguish bonded from fragment contact.
        self.bond_column = {
            (i, j): k
            for i, row in enumerate(family)
            for k, j in enumerate(row)
            if j > i
        }
        devices = [
            d
            for p in cl.get_platforms()
            for d in p.get_devices()
            if d.double_fp_config and "FPGA" not in d.name
        ]
        devices.sort(key=lambda d: not bool(d.type & cl.device_type.GPU))
        if not devices:
            raise RuntimeError("A double-precision OpenCL CPU or GPU is required")
        self.device = devices[0].name
        self.context = cl.Context([devices[0]])
        self.queue = cl.CommandQueue(self.context)

        def buf(array):
            return cl.Buffer(
                self.context,
                cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=np.ascontiguousarray(array),
            )

        self.buf = buf
        self.u = np.zeros((self.n, 3))
        self.v = self.u.copy()
        self.ud, self.vd, self.ad, self.fd, self.extd, self.bodyd = [
            buf(self.u) for _ in range(6)
        ]
        self.refd, self.vold, self.nld = (
            buf(self.reference),
            buf(self.vol),
            buf(self.nl),
        )
        self.zeroi, self.onei = buf(np.zeros((self.n, 3), np.int32)), buf(
            np.ones((self.n, 3), np.int32)
        )
        self.zero, self.density = buf(self.u), buf(np.full((self.n, 3), self.rho))
        self.placeholder = buf(np.zeros(1))
        self.local = [cl.LocalMemory(self.width * 8) for _ in range(3)]
        self.force_program = cl.Program(
            self.context, (source / "peripy/cl/peridynamics.cl").read_text()
        ).build()
        self.step_program = cl.Program(
            self.context, (source / "peripy/cl/euler_cromer.cl").read_text()
        ).build()
        self.cut_program = cl.Program(self.context, CUT_KERNEL).build()
        self.force = self.force_program.bond_force1
        self.step = self.step_program.update_displacement
        self.cut = self.cut_program.knife_cut
        self.stiffness = 12 * self.young / (np.pi * self.horizon**4)
        self.critical_stretch, self.knife_cutting = critical_stretch, knife_cutting
        self.vertices, self.faces, self.planes = blade_mesh()
        self.planed = buf(self.planes)
        self.fragment_pairs = np.empty((0, 2), dtype=int)
        self.max_knife_penetration = self.max_floor_penetration = 0.0

    @property
    def positions(self):
        return self.reference + self.u

    def connectivity(self):
        self.cl.enqueue_copy(self.queue, self.nl, self.nld).wait()
        rows, cols = np.where(self.nl >= 0)
        graph = coo_matrix(
            (np.ones(len(rows)), (rows, self.nl[rows, cols])), shape=(self.n, self.n)
        )
        number, labels = connected_components(graph, directed=False)
        return number, labels

    def update_fragment_contacts(self):
        self.cl.enqueue_copy(self.queue, self.nl, self.nld).wait()
        pairs = cKDTree(self.positions).query_pairs(
            1.1 * self.spacing, output_type="ndarray"
        )
        if not len(pairs):
            self.fragment_pairs = np.empty((0, 2), dtype=int)
            return
        keep = []
        for i, j in pairs:
            column = self.bond_column.get((i, j))
            keep.append(column is None or self.nl[i, column] < 0)
        self.fragment_pairs = pairs[keep]

    def advance(self, center, rotation, velocity, omega):
        cl, q, dt = self.cl, self.queue, self.dt
        x = self.positions
        F = np.zeros_like(x)
        F[:, 2] = -self.mass * 9.81
        # Penalty contact, with dissipative normal damping and Coulomb floor friction.
        stiffness, damping, radius = 100.0, 0.03, 0.45 * self.spacing
        support_z = np.where(
            (np.abs(x[:, 0]) <= 0.18) & (np.abs(x[:, 1]) <= 0.16), 0.0, -0.1
        )
        penetration = np.maximum(support_z + radius - x[:, 2], 0)
        fn = np.maximum(
            stiffness * penetration - damping * np.minimum(self.v[:, 2], 0), 0
        ) * (penetration > 0)
        F[:, 2] += fn
        speed = np.linalg.norm(self.v[:, :2], axis=1)
        ft = np.minimum(0.65 * fn, self.mass * speed / dt)
        F[:, :2] -= ft[:, None] * self.v[:, :2] / np.maximum(speed[:, None], 1e-15)
        self.max_floor_penetration = max(
            self.max_floor_penetration, float(penetration.max())
        )
        pairs = self.fragment_pairs
        if len(pairs):
            a, b = pairs.T
            delta = x[a] - x[b]
            distance = np.linalg.norm(delta, axis=1)
            normal = delta / np.maximum(distance[:, None], 1e-15)
            overlap = 2 * radius - distance
            vn = np.sum((self.v[a] - self.v[b]) * normal, axis=1)
            force = np.maximum(stiffness * overlap - damping * np.minimum(vn, 0), 0) * (
                overlap > 0
            )
            np.add.at(F, a, force[:, None] * normal)
            np.add.at(F, b, -force[:, None] * normal)
        planes = self.planes.copy()
        planes[:, :3] = planes[:, :3] @ rotation.T
        distances = (x - center) @ planes[:, :3].T + planes[:, 3]
        nearest = np.argmax(distances, axis=1)
        gap = distances[np.arange(self.n), nearest] - radius
        normals = planes[nearest, :3]
        tool_velocity = velocity + np.cross(omega, x - center)
        vn = np.sum((self.v - tool_velocity) * normals, axis=1)
        load = np.maximum(-500.0 * gap - 0.06 * np.minimum(vn, 0), 0) * (gap < 0)
        knife_force = load[:, None] * normals
        F += knife_force
        self.max_knife_penetration = max(
            self.max_knife_penetration, float(max(0, -gap.min()))
        )
        cl.enqueue_copy(q, self.extd, np.ascontiguousarray(F / self.vol[:, None]))
        if self.knife_cutting:
            cut_planes = planes.copy()
            # Material points represent finite volumes: use the same tip
            # contact radius for the cutting front and the repulsive contact.
            tip = self.planes[:, 2] < -0.9
            cut_planes[tip, 3] -= radius
            cl.enqueue_copy(q, self.planed, np.ascontiguousarray(cut_planes))
            self.cut(
                q,
                (self.n, self.width),
                None,
                self.refd,
                self.ud,
                self.nld,
                self.planed,
                np.int32(self.width),
                np.int32(len(planes)),
                *map(np.float64, center),
                *map(np.float64, rotation[:, 0]),
            )
        self.force(
            q,
            (self.n * self.width,),
            (self.width,),
            self.ud,
            self.fd,
            self.bodyd,
            self.refd,
            self.vold,
            self.nld,
            self.onei,
            self.extd,
            self.placeholder,
            self.placeholder,
            self.placeholder,
            self.placeholder,
            *self.local,
            np.float64(self.stiffness),
            np.float64(self.critical_stretch),
            np.float64(1),
            np.int32(1),
        )
        self.step(
            q,
            (self.n * 3,),
            None,
            self.fd,
            self.ud,
            self.vd,
            self.ad,
            self.zeroi,
            self.zero,
            self.density,
            np.float64(0),
            np.float64(self.rho * 8),
            np.float64(dt),
        )
        cl.enqueue_copy(q, self.u, self.ud)
        cl.enqueue_copy(q, self.v, self.vd).wait()
        return -knife_force.sum(axis=0), -np.cross(x - center, knife_force).sum(axis=0)


def run_cutting(
    output,
    *,
    source=None,
    spacing=0.0025,
    dt=5e-5,
    duration_s=4.2,
    coupling_dt=1 / 120,
    critical_stretch=0.5,
    knife_cutting=True,
    cut_positions=(0.045, 0.0, -0.045),
    fps=20,
):
    import pybullet as p
    import pybullet_data
    from .robot_tools import knife_robot_urdf

    if (
        not 0.0015 <= spacing <= 0.004
        or not 0 < dt <= 1e-4
        or not 0 < coupling_dt <= 1 / 120
    ):
        raise ValueError(
            "require spacing 1.5–4 mm, dt <= 100 us and coupling dt <=1/120 s"
        )
    if (
        not 1 <= fps <= 30
        or not 0 < duration_s <= len(cut_positions) * 1.4 + 1e-12
        or not 0 < critical_stretch
        or not np.isfinite(critical_stretch)
    ):
        raise ValueError("invalid frame rate, duration or critical stretch")
    source = Path(source or os.environ.get("SOFTGRIPLAB_PERIPY", ""))
    revision = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
        cwd=source,
        text=True,
    ).strip()
    if revision != PERIPY_COMMIT:
        raise ValueError(f"Use PeriPy revision {PERIPY_COMMIT}; see docs/CUTTING.md")
    for name, expected in KERNEL_SHA256.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"PeriPy kernel differs from the pinned source: {name}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    substeps = int(np.ceil(coupling_dt / dt))
    model = PeriPyMaterial(
        source, spacing, coupling_dt / substeps, critical_stretch, knife_cutting
    )
    print(
        f"PeriPy: {model.n} material points, {model.initial_count//2} bonds, {model.device}",
        flush=True,
    )
    # Gravity equilibration is real integration, with the blade clear of the body.
    for tick in range(round(0.4 / model.dt)):
        model.advance(np.array([0.0, 0.0, 0.2]), np.eye(3), np.zeros(3), np.zeros(3))
    initial_pieces, _ = model.connectivity()
    if initial_pieces != 1 or (model.nl >= 0).sum() != model.initial_count:
        raise RuntimeError("settling damaged the initially intact material")
    settled = model.positions.copy()
    cid = p.connect(p.DIRECT)
    snapshots, times, robot_states, poses, labels_history, rows = [], [], [], [], [], []
    reaction, torque = np.zeros(3), np.zeros(3)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(coupling_dt, physicsClientId=cid)
        urdf, mount = knife_robot_urdf(
            Path(pybullet_data.getDataPath()) / "franka_panda",
            output,
            model.vertices,
            model.faces,
            world_quaternion=(0, 0, 0, 1),
        )
        robot = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=cid)
        orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        for j, angle in enumerate([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8]):
            p.resetJointState(robot, j, angle, physicsClientId=cid)

        def command(t, initial=False):
            target = tool_target(t, cut_positions) + OFFSET + [0, 0, 0.03]
            ik = p.calculateInverseKinematics(
                robot,
                11,
                target,
                orientation,
                maxNumIterations=300 if initial else 80,
                residualThreshold=1e-7,
                physicsClientId=cid,
            )
            if initial:
                for j in range(7):
                    p.resetJointState(robot, j, ik[j], physicsClientId=cid)
            p.setJointMotorControlArray(
                robot,
                list(range(7)),
                p.POSITION_CONTROL,
                targetPositions=ik[:7],
                forces=[120.0] * 7,
                positionGains=[1 - 0.82 ** (coupling_dt * 120)] * 7,
                velocityGains=[1.0] * 7,
                physicsClientId=cid,
            )

        def measured():
            link = p.getLinkState(
                robot, 11, computeForwardKinematics=True, physicsClientId=cid
            )
            center, quat = p.multiplyTransforms(link[4], link[5], [0, 0, 0.03], mount)
            return np.asarray(center) - OFFSET, np.asarray(quat)

        command(0, initial=True)
        center, quat = measured()
        steps, stride = round(duration_s / coupling_dt), max(
            1, round(1 / (fps * coupling_dt))
        )
        for tick in range(steps + 1):
            t = tick * coupling_dt
            if tick:
                command(t)
                p.applyExternalForce(
                    robot,
                    11,
                    reaction,
                    center + OFFSET,
                    p.WORLD_FRAME,
                    physicsClientId=cid,
                )
                p.applyExternalTorque(
                    robot, 11, torque, p.WORLD_FRAME, physicsClientId=cid
                )
                p.stepSimulation(physicsClientId=cid)
                end, end_q = measured()
                if np.dot(end_q, quat) < 0:
                    end_q *= -1
                velocity = (end - center) / coupling_dt
                omega = (
                    Rotation.from_quat(end_q) * Rotation.from_quat(quat).inv()
                ).as_rotvec() / coupling_dt
                reaction, torque = np.zeros(3), np.zeros(3)
                model.update_fragment_contacts()
                for sub in range(substeps):
                    fraction = (sub + 0.5) / substeps
                    qmid = quat + (end_q - quat) * fraction
                    qmid /= np.linalg.norm(qmid)
                    force, moment = model.advance(
                        center + (end - center) * fraction,
                        Rotation.from_quat(qmid).as_matrix(),
                        velocity,
                        omega,
                    )
                    reaction += force / substeps
                    torque += moment / substeps
                center, quat = end, end_q
            number, labels = model.connectivity()
            broken = 1 - (model.nl >= 0).sum() / model.initial_count
            rows.append(
                [
                    t,
                    *center,
                    *reaction,
                    *torque,
                    broken,
                    number,
                    np.linalg.norm(center - tool_target(t, cut_positions)),
                    float(model.positions[:, 2].min()),
                    float(np.linalg.norm(model.v, axis=1).max()),
                ]
            )
            if (
                not np.isfinite(model.u).all()
                or np.max(abs(model.u)) > 2
                or np.max(np.linalg.norm(model.v, axis=1)) > 20
            ):
                np.savez_compressed(
                    output / "failure_state.npz",
                    time_s=t,
                    position_m=model.positions,
                    velocity_m_s=model.v,
                    reference_position_m=model.reference,
                    neighbours=model.nl,
                    labels=labels,
                    knife_center=center,
                    knife_rotation=quat,
                )
                raise RuntimeError(f"unstable cutting state at {t:g} s")
            if tick % stride == 0 or tick == steps:
                snapshots.append(model.positions.copy())
                times.append(t)
                robot_states.append(
                    [
                        p.getJointState(robot, j, physicsClientId=cid)[0]
                        for j in range(12)
                    ]
                )
                poses.append(np.r_[center, quat])
                labels_history.append(labels.copy())
            if tick % 60 == 0:
                print(
                    f"Cutting {t:.2f} s: pieces {sorted(np.bincount(labels), reverse=True)[:8]}, damage {broken:.4f}, force {reaction[2]:.3f} N",
                    flush=True,
                )
        np.savez_compressed(
            output / "trajectory.npz",
            time_s=times,
            position_m=snapshots,
            robot_joint_position=robot_states,
            knife_pose_local=poses,
            knife_vertices=model.vertices,
            knife_faces=model.faces,
            world_offset_m=OFFSET,
            reference_position_m=model.reference,
            settled_position_m=settled,
            component_labels=labels_history,
            particle_volume_m3=model.vol,
            spacing_m=spacing,
            initial_neighbours=model.initial,
            final_neighbours=model.nl,
        )
        np.savetxt(
            output / "observables.csv",
            rows,
            delimiter=",",
            comments="",
            header="time_s,knife_x_m,knife_y_m,knife_z_m,reaction_x_n,reaction_y_n,reaction_z_n,torque_x_nm,torque_y_nm,torque_z_nm,broken_fraction,connected_components,tracking_error_m,bottom_z_m,max_speed_m_s",
        )
        sizes = np.bincount(labels)
        major = sizes > model.n * 0.03
        summary = dict(
            backend="PeriPy OpenCL + PyBullet Panda",
            peripy_commit=PERIPY_COMMIT,
            opencl_device=model.device,
            particles=model.n,
            spacing_m=spacing,
            horizon_m=model.horizon,
            young_pa=model.young,
            poisson=0.25,
            density_kg_m3=model.rho,
            mass_kg=float(model.mass.sum()),
            dt_s=model.dt,
            coupling_dt_s=coupling_dt,
            duration_s=duration_s,
            cut_positions_m=list(map(float, cut_positions)),
            settling_duration_s=0.4,
            critical_stretch=critical_stretch,
            geometric_knife_cutting=knife_cutting,
            preset_interfaces=0,
            initial_broken_bonds=0,
            initial_connected_components=initial_pieces,
            final_connected_components=number,
            final_component_sizes=sorted(np.bincount(labels).tolist(), reverse=True),
            final_major_components=int(major.sum()),
            numerical_debris_mass_fraction=float(sizes[~major].sum() / model.n),
            maximum_particle_speed_m_s=float(np.asarray(rows)[:, 14].max()),
            max_floor_penetration_m=model.max_floor_penetration,
            max_knife_penetration_m=model.max_knife_penetration,
            reaction_force_and_torque_feedback=True,
            clamped_nodes=0,
            success=bool(
                int(major.sum()) == len(cut_positions) + 1
                and sizes[~major].sum() / model.n < 0.01
            ),
            backend_source_sha256={
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ["peripy/cl/peridynamics.cl", "peripy/cl/euler_cromer.cl"]
            },
            limitations=[
                "Geometric severing of bonds intersected by the measured finite blade; not a calibrated fracture-energy model.",
                "Bond-based microelastic bulk with tensile critical-stretch failure; fixed Poisson ratio 0.25.",
                "Homogeneous uncalibrated fruit surrogate without peel or fibres.",
                "Penalty floor/blade/fragment contact; fragment neighbour search once per coupling step.",
                "First-order partitioned robot coupling; robot links outside the blade do not contact material.",
            ],
        )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        p.disconnect(physicsClientId=cid)
