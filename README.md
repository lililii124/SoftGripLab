# SoftGripLab

**Learn the material. Understand the contact. Make the grasp work.**

An animation-first mechanics toolkit for material identification, adhesive
contact, soft-object grasping and robot suction demonstrations. Built on
**SciPy**, **JAX-MPM** and **PyBullet**, with their existing solvers and robot assets.

[中文说明](README.zh-CN.md) · [Reproduce the animations](docs/ANIMATIONS.md) ·
[Models](docs/MODELS.md) · [Validation](docs/VALIDATION.md)

## 1 · Press a specimen and identify its response

![Fast and slow compression, animated from synthetic records](docs/assets/01_compression.gif)

Two loading rates reveal the same material's relaxation behavior. The platens
follow the recorded displacement; force changes are shown directly beside the
specimen. SciPy jointly fits the records and checks an unseen loading history.
**The geometry is a uniform-strain illustration driven by synthetic data.**

## 2 · Add adhesion: touch, retract, pull off

![Hertz and JKR contact during approach and retraction](docs/assets/02_adhesion.gif)

Hertz contact opens on separation. JKR contact can remain attached and carry
tension during retraction, then detaches at its displacement-controlled
instability. Contact radius, force and attachment history come directly from
JAX-MPM's existing contact laws. **The local surface shape is schematic; this is
dry surface adhesion, not vacuum suction.**

## 3 · Grip a deformable block and challenge the contact

![Side-by-side simulated soft blocks with and without jaw friction](docs/assets/03_grasp.gif)

The same 3D MPM block is squeezed, loaded by gravity and subjected to a base
acceleration pulse. The animation compares frictionless and frictional jaws;
color shows computed deformation. **These are sections of actual simulated
particle trajectories.** Coarse grid contact still has spatial discretization
error; the scene is a benchmark, not a calibrated physical gripper.

## 4 · Move a robot: seal, lift, transfer, release

![Panda robot performing a suction pick-and-place, with tool close-up](docs/assets/04_robot.gif)

The existing Panda URDF is driven by PyBullet joint motors and inverse kinematics.
The suction tool only attaches after detected contact, footprint coverage and
alignment checks. Its pressure-area capacity can be exceeded, opening the joint.
The robot transports the object to the green target and releases it; a weak-vacuum
control fails to lift it. **The payload is rigid, the suction joint is a reduced
model, and this scene is not co-simulated with the separate soft-material model.**

## Run a demonstration

Reference models and compression animation, Python 3.11+:

```bash
python -m pip install '.[test]'
python -m softgriplab demo --output outputs/demo
python -m softgriplab animate-compression
python -m pytest -q
```

Robot demo, optional PyBullet dependency; CPU rendering works without a display:

```bash
python -m pip install '.[robot]'
python -m softgriplab robot-demo --output outputs/robot
python -m softgriplab robot-demo --pressure-pa 1000 --no-render --output outputs/robot_weak
```

Adhesion and 3D soft-body scenes use the existing JAX-MPM backend in Python 3.12+:

```bash
python -m pip install '.[mpm]'
python -m softgriplab adhesion --output outputs/adhesion
python -m softgriplab mpm-grasp --output outputs/mpm
python -m softgriplab mpm-grasp --friction 0 --output outputs/mpm_frictionless
python -m softgriplab animate-grasp
```

Animation commands write a looping GIF and a PNG preview. Simulations also export
CSV traces and JSON settings/results. The MPM scene exports VTK for ParaView.
Use separate environments if your robot and JAX installations require different
Python versions. No commercial software, GPU or hardware is required by these demos.

## Identify your own material

CSV header (SI units):

```csv
time_s,compression_m,force_n
0,0,0
0.01,0.00001,0.02
0.02,0.00002,0.038
```

Use complete ramp/hold records, not these three illustrative rows. Time starts
at zero and must be uniformly sampled. Compression means total thickness
reduction; force is measured on **one** opposing platen. Preprocess instrument
offsets explicitly; the reference model assumes zero prehistory and uniform
uniaxial strain. General indentation requires a geometry-aware forward model.

```bash
python -m softgriplab identify fast.csv slow.csv \
  --thickness-m 0.02 --area-m2 0.0004 --noise-std-n 0.005 \
  --output outputs/material.json
```

Fits report equilibrium/branch moduli, relaxation time, residual error, Jacobian
conditioning and local uncertainty. Friction is an independent input. The
existing fixed-closure, ideal-force and identified-closure comparisons remain
available through `demo`; their detailed plots are supplementary validation,
not the main gallery.

## Reuse and scope

| Existing tool | What it supplies |
| --- | --- |
| SciPy | State-space integration and bounded material identification |
| JAX-MPM | Continuum solver, contact boundaries, Hertz/JKR history and VTK export |
| PyBullet / pybullet_data | Robot dynamics, IK, collision detection, Panda URDF and CPU camera |
| Pillow / Matplotlib | Data-driven animation, labels and supplementary figures |
| pytest / GitHub Actions | Analytic checks, inference tests and robot/backend integration tests |

The four stages deliberately expose different model levels. Viscoelastic
identification, elastic MPM, JKR and rigid robot suction are **not yet one coupled
simulator**. There is no hardware validation, learned control or calibrated
damage prediction in this release. [Roadmap](docs/ROADMAP.md).

[GPL-3.0](LICENSE) · [Third-party attribution](NOTICE). Upstream solvers and robot
mesh files are loaded from their dependencies, not copied into this repository.
