# Reproduce the simulation gallery

All four primary animations use solver geometry. Surface rendering runs after
FEM/MPM simulation, reads saved node coordinates, and uses a depth buffer. It
never rescales deformation or prescribes a specimen shape. The robot camera
captures the actual PyBullet state, with a simultaneous finger close-up.

| Scene | Solver and contact | Saved geometry |
| --- | --- | --- |
| Compression | Bullet Neo-Hookean tetrahedral FEM; finite platens | Actual nodes, tetrahedra and boundary triangles |
| Adhesion | JAX-MPM + upstream Dugdale surface traction | Advected particle surface and sphere position |
| Soft grasp | Bullet FEM; finite jaws, friction and gravity | Actual nodes and fixture positions; no anchors |
| Panda | PyBullet articulated rigid dynamics; finger collision and friction | Joint states, payload pose and measured contacts |

## Generate the gallery

```bash
# Python 3.11+; FEM, robot, and offline rendering
python -m pip install '.[robot,test]'
python -m softgriplab fem-demo --output outputs/fem_compression
python -m softgriplab fem-demo --scene grasp --output outputs/fem_grasp
python -m softgriplab robot-demo --output outputs/robot
python -m softgriplab robot-demo --friction 0 --no-render --output outputs/robot_frictionless

# Python 3.12+; pinned upstream MPM backend
python -m pip install '.[mpm]'
python -m softgriplab mpm-adhesion --output outputs/continuum_adhesion
python -m softgriplab mpm-adhesion --work 0 --output outputs/continuum_nonadhesive

# In an environment containing PyBullet, after both MPM trajectories exist
python -m softgriplab animate-adhesion
```

The dense gallery uses 9,504 tetrahedra per FEM specimen and 45,968 material
points per surface-adhesion case. Its output folders and collector are:

```bash
python -m softgriplab fem-demo --output outputs/dense_compression
python -m softgriplab fem-demo --scene grasp --output outputs/dense_grasp
python -m softgriplab fem-demo --scene grasp --friction 0 --no-render --output outputs/dense_grasp_zero
python -m softgriplab mpm-adhesion --output outputs/dense_adhesion_14.4
python -m softgriplab mpm-adhesion --work 0 --output outputs/dense_adhesion_0
python -m softgriplab animate-adhesion --source outputs/dense_adhesion_14.4 --comparison outputs/dense_adhesion_0 --output outputs/gallery/dense_adhesion.gif
python -m softgriplab robot-demo --output outputs/robot_minimal
python tools/collect_dense_gallery.py
```

The earlier temporal comparison is preserved under `legacy_refinement/`; the
new spatial comparison uses equal 0.5 ms steps on 1,764 and 9,504 tetrahedra.
This is a sensitivity check, not a claim of spatial convergence.

The collector checks contact outcomes, deformation validity and render input
hashes before replacing the gallery. It copies summaries, CSV observations and
compact surface replay files to `docs/results/animations/`. Replay files contain
the unscaled surface coordinates used to render, along with topology and time.
Full solver states remain in `outputs/`; the MPM surface replay omits interior
particles and their fields to keep the repository small.

Replay a committed surface without rerunning its solver:

```bash
python tools/replay_surface.py docs/results/animations/fem_grasp_surface.npz --output outputs/replay.gif
```

This produces the 3D surface view. Interior FEM section nodes are present in the
full trajectory, but are omitted from the compact surface-only replay.

## Rendering and interpretation

`fem-demo --no-render` only solves; `animate-compression --source PATH` subsequently
renders either a compression or grasp FEM directory. The legacy command name is
retained for compatibility, but it no longer draws synthetic uniform strain.
`animate-adhesion` renders the two saved continuum trajectories without importing
JAX. The robot uses PyBullet's existing Panda meshes and CPU camera.

Frame times and parameters are stored in the CSV/JSON files. Animations retain
only short labels: 3 s of FEM motion plays over about 4.5 s;
0.13 s of MPM pull-off plays over about 5 s to expose the contact sequence.
Robot playback is approximately real time. Robot trace timestamps describe the
state after each integration step. Grey tools and muted rubber colors identify
materials; they do not encode strain or damage. No rainbow field is implied.

The MPM display connects the outside material sampling layers. Their initial
surface lies half a particle spacing inside the prescribed continuum block;
this is also the surface used by the adhesive interaction. The finite attraction
range is 2.5 mm. It is a coarse dry-adhesion demonstration, not a resolved vacuum
cup seal, analytical JKR solution, or converged pull-off prediction.

FEM tools are prescribed kinematic boundaries. A zero-mass Bullet fixture is
advanced from its previous pose with the corresponding velocity; saved poses
are read back from Bullet and checked against the intended boundary condition.
This supplies both the normal displacement and tangential velocity required by
friction. Specimen nodes are never reset or attached to tools.

The robot uses joint motors after its initial configuration. Both finger contact
forces must remain positive while the cube is airborne. Payload collisions stay
enabled and there is no payload attachment constraint. A 0-friction control
must fail to lift. The robot object is rigid and is separate from the FEM model.

## Earlier command changes

`robot-demo --pressure-pa` was replaced by `--friction` and `--finger-force-n`.
The earlier suction-joint demonstration and schematic deformation animations
were removed. `adhesion` now exports only the numerical Hertz/JKR reference CSV;
use `mpm-adhesion` and `animate-adhesion` for resolved deformation.

Upstream implementations: [Bullet FEM example](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/deformable_torus.py),
[JAX-MPM](https://github.com/lililii124/jax-mpm),
[Panda asset](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/franka_panda).

## Sequential robot cutting

The sixth animation replays the solved FEM and cut-face geometry from three
Panda knife strokes. [Setup, model and controls](CUTTING.md). To replay the
committed data with PyBullet only:

```bash
python -m softgriplab animate-cutting --source docs/results/cutting/nominal
```
