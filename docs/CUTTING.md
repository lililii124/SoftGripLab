# Robot cutting without predefined interfaces

The initial fruit surrogate is one connected body with no cut faces, duplicated
nodes or weakened planes. The Panda moves a finite wedge-shaped knife through
it. Blade contact and bond severing use the measured robot pose during integration.
The tool path can change without rebuilding the initial material connectivity.

## Reused mechanics

[PeriPy](https://github.com/alan-turing-institute/PeriPy), revision
`04cae072e22568e29b9ab8f84d821f9576ecb728`, supplies the unmodified OpenCL
`bond_force1` and Euler–Cromer kernels. They integrate microelastic bond forces,
tensile critical-stretch failure, gravity and externally supplied contact loads.
PeriPy is MIT licensed. SoftGripLab adds the moving-tool adapter.

During each substep, a material bond segment is tested against the current finite
blade mid-surface, bounded by the moving blade mesh. Only bonds crossing
from one blade side to the other are removed. The front uses the same h × 0.45
material-point contact radius as repulsion; it is not a preinserted interface. The initial bond list
has no knowledge of the tool path. This is geometric cutting with solved bulk
dynamics; it does not predict cutting force from calibrated fracture toughness.

The robot receives the interval-averaged material force and torque. Its measured
position and orientation drive contact. The visible knife uses the same mesh and
transform as the contact calculation.

| Quantity | Value |
| --- | --- |
| Geometry | Curved 160 mm body; homogeneous surrogate; no peel or fibres |
| Material-point spacing | 2.5 mm |
| Nonlocal horizon | 2.51 times the spacing |
| Bulk | E = 50 kPa; density 1,000 kg/m³; bond-based nu = 0.25 |
| Tensile failure | Critical stretch 0.5, additional to blade-intersection cutting |
| Initial condition | 0.4 s of solved gravity settling before the first displayed frame |
| Support | Finite board; friction coefficient 0.65; floor below the board |
| Constraints | No anchors or clamped end |

Separated material retains position, velocity and mass. Short-range penalty
contact acts between disconnected material samples. Its neighbour search updates
once per coupling interval. Isolated numerical debris remains in the archive and
mass accounting; major pieces and debris mass are reported separately. Neither
debris nor cutting force is a calibrated prediction for real banana tissue.

## Rendering and records

scikit-image marching cubes reconstructs the surface from saved particle positions.
Particle volume is deposited on voxels of spacing h/3, smoothed with a Gaussian
of width 0.75 h, and contoured at density 0.4 for each solved connected component.
The reconstruction has no input describing the knife path or expected cuts.
Sparse fragments below the density isovalue are shown as material samples with
their contact radius; no connected component is omitted from rendering.
Deformation scale is 1:1. This stated surface reconstruction does not refine the
mechanical discretization.

Robot joints, tool transforms, all material points, initial/final bonds, component
labels and particle volumes are saved in `trajectory.npz`. JSON/CSV records retain
parameters and observations; the animation has no text overlays. A synchronized inset shows the recorded
vertical reaction on the knife. It is a contact reaction, not a fracture-energy
calibrated cutting-force prediction.

## Setup and reproduction

```bash
python -m pip install '.[cutting]'
python tools/setup_peripy.py ../peripy-backend
python -m softgriplab banana-cutting --backend ../peripy-backend --output outputs/banana_robot
python -m softgriplab animate-cutting --source outputs/banana_robot
```

The solver needs a double-precision OpenCL CPU or GPU device and its runtime.
Only PeriPy source kernels are loaded; its Python/Cython and HDF5 dependencies
are not required. Replay needs PyBullet and scikit-image, without OpenCL.

A changed path uses identical initial connectivity. An intact-body control
disables blade severing and suppresses stretch failure:

```bash
python -m softgriplab banana-cutting --backend ../peripy-backend --cut-x .035 --duration 1.4 --output outputs/pd_shifted
python -m softgriplab banana-cutting --backend ../peripy-backend --cut-x .045 --no-knife-severing --critical-stretch 1000000 --duration .65 --output outputs/pd_intact
```

The prior DiSECt experiment is a [separate legacy example](CUTTING_DISECT.md),
run with `banana-cutting-disect` under its NVIDIA research license. The current
PeriPy scene does not import DiSECt or use its knife geometry.

## Archived validation

The [saved runs](results/dynamic_cutting/) include the complete three-stroke case,
a shifted first stroke using the identical initial graph, and a contacted intact
control with both severing and stretch failure disabled. Both tested time steps
produce four major pieces. Small numerical fragments remain: about 0.30% of mass
at the coarser step and 0.90% at the finer step. All material samples are retained
and rendered. This sensitivity prevents a claim of converged fragmentation or
cutting force; the F–t inset is the recorded contact reaction of this model.

```bash
python -m softgriplab banana-cutting --backend ../peripy-backend --output outputs/pd_nominal
python -m softgriplab banana-cutting --backend ../peripy-backend --dt .0001 --no-render --output outputs/pd_ductile
# Also run the shifted and intact controls above.
python tools/collect_dynamic_cutting.py
```
