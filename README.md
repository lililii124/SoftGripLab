# SoftGripLab

Contact and material mechanics for robotic grasping, adhesion, and cutting.
Uses **Bullet FEM**, **JAX-MPM**, **SciPy** and the existing **Franka Panda** model.

[中文说明](README.zh-CN.md) · [Reproduce the animations](docs/ANIMATIONS.md) ·
[Models](docs/MODELS.md) · [Validation](docs/VALIDATION.md)

## 1 · Compress an elastic specimen

![Solved tetrahedral specimen during compression, hold and release](docs/assets/01_compression.gif)

A 50 × 50 × 40 mm Neo-Hookean block compresses, bulges sideways and recovers.
Bullet solves **9,504 tetrahedra**; the renderer reads their saved boundary nodes.
The section on the right follows the same material nodes. Deformation is shown
at **1:1 scale**. This elastic example is separate from the viscoelastic fit below.

## 2 · Indent, adhere and pull off

![Actual MPM surface deformation with and without adhesion](docs/assets/02_adhesion.gif)

The same sphere trajectory acts on two 3D MPM layers. One has no adhesion; the
other receives finite-range surface tractions from the upstream Dugdale law.
Those forces act on the **current, deformed surface** and enter the bulk solver.
The surface rises during withdrawal and then detaches. This is dry adhesion;
it does not model vacuum sealing or leakage.

## 3 · Grip, lift and release a soft block

![Finite jaws physically contacting and lifting a solved soft block](docs/assets/03_grasp.gif)

Two finite jaws squeeze the tetrahedral block, lift it through friction, and
open so it falls back onto the support. No nodes are attached to the jaws.
A zero-friction control fails to lift the specimen. Tool geometry and the
specimen surface are rendered from the saved simulation state.

## 4 · Pick and place with the Panda

![Panda bilateral finger contact with a live close-up and measured contact forces](docs/assets/04_robot.gif)

Joint motors close the fingers on a **rigid** cube and move it to the target.
The close-up shows the actual finger contact. Forces and signed contact distances
are recorded in CSV. There is no payload attachment joint or disabled finger collision.
A frictionless control fails to lift the cube. This articulated robot example
is separate from the deformable fixture example above.

## 5 · Carry a deformable object by adhesion

![Panda adhesive grasp with the actual deformable material surface](docs/assets/05_adhesive_robot.gif)

The Panda presses a spherical adhesive head onto a **27 g deformable block**,
lifts it, carries it 80 mm and switches adhesion off after setting it down.
Measured robot motion drives the MPM contact calculation; the material reaction
loads the arm on the next coupling step. The object has gravity and no anchors
or attachment joints. A zero-adhesion control remains on the table.

The close-up replays the saved material mesh and measured robot joints at 1:1
deformation scale. [Model, checks and replay data](docs/ADHESIVE_ROBOT.md).

## 6 · Make three cuts with the Panda

![Panda cutting the solved banana-shaped material into four connected components](docs/assets/06_banana_cutting.gif)

The robot cuts a curved fruit surrogate three times. The **FEM mesh deforms under
knife contact**, cohesive springs break, and connectivity changes from one body
to four pieces. Joint motion and material reaction are coupled; each stroke
continues from the previous material state. The close-up uses saved solver nodes.

This uses an optional **DiSECt research backend** with a separate NVIDIA license.
Material parameters are illustrative and the three fracture planes are planned
in advance. [Model, controls, license and reproduction](docs/CUTTING.md).

## Run the scenes

Python 3.11+, CPU only:

```bash
python -m pip install '.[robot,test]'
python -m softgriplab fem-demo --output outputs/fem_compression
python -m softgriplab fem-demo --scene grasp --output outputs/fem_grasp
python -m softgriplab robot-demo --output outputs/robot
python -m softgriplab robot-demo --friction 0 --no-render --output outputs/robot_frictionless
python -m pytest -q
```

Adhesion uses the pinned JAX-MPM backend, Python 3.12+:

```bash
python -m pip install '.[mpm,robot]'
python -m softgriplab mpm-adhesion --output outputs/continuum_adhesion
python -m softgriplab mpm-adhesion --work 0 --output outputs/continuum_nonadhesive
python -m softgriplab animate-adhesion
python -m softgriplab adhesive-robot --output outputs/adhesive_robot
python -m softgriplab adhesive-robot --work 0 --no-render --output outputs/adhesive_robot_control
```

You can run the solver and renderer in separate environments: `animate-adhesion`
needs PyBullet and the saved NPZ files, without importing JAX. FEM also supports
`--no-render`; `animate-compression --source outputs/fem_compression` renders the
saved trajectory. No physics is recomputed during mesh rendering.

Each scene exports GIF/PNG, simulation settings and numerical observations.
The soft-body scenes retain raw coordinates and topology in `trajectory.npz`;
FEM also writes its input OBJ and tetrahedral VTK mesh. [Detailed commands](docs/ANIMATIONS.md).

## Identify material response

The SciPy reference workflow fits a standard linear solid to multiple ramp/hold
records, checks an unseen loading history, and compares grasp-force baselines.
The bundled records are synthetic. It is independent of the elastic scene above.

```bash
python -m softgriplab demo --output outputs/demo
python -m softgriplab identify fast.csv slow.csv \
  --thickness-m 0.02 --area-m2 0.0004 --noise-std-n 0.005 \
  --output outputs/material.json
```

Input CSV columns: `time_s,compression_m,force_n`, in SI units. Use complete,
uniformly sampled ramp/hold records starting at zero. Compression is total
thickness reduction; force is measured on **one** opposing platen. Remove sensor
offsets before fitting. Fits report residuals, conditioning and local uncertainty.

The original planar-jaw MPM disturbance benchmark remains available through
`mpm-grasp`; `animate-grasp` renders its particle-section comparison. Hertz/JKR
reference histories remain available as CSV through `adhesion`.

## Reuse and limits

| Dependency | Role |
| --- | --- |
| Bullet / PyBullet | Neo-Hookean tetrahedral FEM, finite contact, robot dynamics and depth-buffered rendering |
| JAX-MPM | Bulk continuum dynamics, Dugdale surface traction, Hertz/JKR references, planar-jaw shock case |
| SciPy | State-space integration and bounded material identification |
| DiSECt (optional, research license) | Virtual-node FEM cutting, knife contact and cohesive damage |
| pybullet_data | Panda URDF and robot meshes |
| NumPy / Pillow / Matplotlib | Data processing, animation export and supplementary plots |

The contribution is the benchmark setup, solver adapters, observable contact
checks and reproducible comparison workflow. It does not implement a new physics
engine. Spatial convergence, experimental calibration and transfer of the fitted
SLS law into the continuum scenes remain open work. Scene 4 uses a rigid cube;
scene 5 couples the Panda to an elastic MPM object; scene 6 adds planned sequential
cutting of an uncalibrated fruit surrogate. [Roadmap](docs/ROADMAP.md).

[GPL-3.0](LICENSE) · [Third-party attribution](NOTICE).
