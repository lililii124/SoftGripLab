# Animation gallery and reproduction

The main gallery goes from a material record to a contact law, a deformable
grasp, and an articulated robot. Every animation is linked to its data source.
Numbers and force indicators accompany the scene; standalone curves live in
the supplementary validation files.

| Stage | Data / engine | Geometry in the animation |
| --- | --- | --- |
| 01 Compression | Saved synthetic ramp/hold CSVs | Uniform-strain specimen illustration, constant loaded area |
| 02 Adhesion | Upstream Hertz/JKR displacement-controlled contact history | Local surface sketch; radius, force and attachment are computed |
| 03 Grasp | Existing 3D Neo-Hookean MPM trajectories | Central section of actual particle coordinates, colored by Green strain norm |
| 04 Robot | PyBullet Panda URDF, motor control and rigid-body dynamics | Two simulator cameras, full robot and live tool/payload close-up |

## Commands

```bash
python -m softgriplab demo --output outputs/demo
python -m softgriplab animate-compression --source outputs/demo

# Python 3.12+, optional JAX-MPM backend
python -m softgriplab adhesion --output outputs/adhesion
python -m softgriplab mpm-grasp --output outputs/mpm
python -m softgriplab mpm-grasp --friction 0 --output outputs/mpm_frictionless
python -m softgriplab animate-grasp --source outputs/mpm --comparison outputs/mpm_frictionless

# Optional PyBullet backend
python -m softgriplab robot-demo --output outputs/robot
python -m softgriplab robot-demo --pressure-pa 1000 --no-render --output outputs/robot_weak

# After generating the four gallery inputs above
python tools/collect_animations.py
```

The collector copies GIFs and preview images to `docs/assets/` and stores
simulation summaries and contact/robot CSVs under `docs/results/animations/`.
It verifies each GIF has multiple frames and checks the nominal/weak robot
outcomes and adhesive detachment before updating the gallery manifest.

The time labels show the source/simulation time. GIF playback can be slowed to
make fast dynamics visible. Rendering does not advance the physics. In the robot
log, the time label is the control-step start and the state is captured after
that step (1/240 s). Camera settings and overlays do not affect the dynamics.

## Adhesion is not suction

JKR describes a small dry adhesive sphere/half-space contact using work of
adhesion and reduced elastic modulus. This demo carries attachment state during
approach and retraction and holds briefly in the tensile regime. It does not
model a vacuum chamber or leakage.

The robot demo uses a reduced vacuum-gripper joint. A detected contact,
approximately aligned surface normal and full cup footprint are required to
acquire a seal. A fixed trial constraint supplies the demand wrench. It breaks
if the conservative force norm exceeds `efficiency * pressure * pi * radius²`
or the moment norm exceeds that force times half the cup radius. Thus even a
large compressive preload can break this conservative approximation; it is not
a full anisotropic suction/contact law. The weaker-pressure run breaks during
the contact/preload phase, not after a successful airborne lift.

The trial constraint is removed on the next control step, allowing a one-step
capacity overshoot. No compliant cup, fluid dynamics, leakage, inferred seal
quality or deformable payload is simulated. The Panda path is scripted; inverse
kinematics and joint motors execute it. This is a robot demonstration with
explicit model limits, not a hardware-validated grasp policy.

## Keep source and model attribution visible

Do not present the homogeneous specimen sketch or JKR surface sketch as a
resolved displacement field. Do not treat the separate MPM and robot scenes as
one coupled model. The renderer loads the upstream Panda asset through
`pybullet_data`; the robot is not hand-drawn or animated through resets.

Upstream: [JAX-MPM](https://github.com/lililii124/jax-mpm),
[PyBullet](https://github.com/bulletphysics/bullet3),
[Panda asset](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/franka_panda).
