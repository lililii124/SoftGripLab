# Panda sequential cutting

![Three cuts through the solved fruit mesh](assets/06_banana_cutting.gif)

The Panda makes three strokes through a curved, homogeneous peeled-fruit
surrogate. The saved FEM surface deforms under knife contact, each cohesive
interface loses stiffness, and the material connectivity changes from one
component to four. The end band stays clamped. Separated pieces retain their
positions and velocities; the adapter never resets or moves them into place.

## Reused solver and license

The optional backend is [NVIDIA DiSECt](https://github.com/NVlabs/DiSECt), commit
`ff90165070335bb8033000373d2d20c6aac13edd`, described in the
[DiSECt project](https://diff-cutting-sim.github.io/). It supplies the FEM,
knife contact, virtual-node cut topology and damage kernels. PyBullet supplies
Panda dynamics and rendering. SoftGripLab supplies the specimen, robot adapter,
cut-volume integration, checks and replay workflow.

**DiSECt has a separate NVIDIA research/evaluation license.** It is not covered
by SoftGripLab's GPL and is not a permissively licensed dependency. Its source
and binaries are not bundled in this repository. Read its
[license](https://github.com/NVlabs/DiSECt/blob/main/LICENSE.md) before reuse.
The default installation and archived-mesh replay do not import DiSECt.

## Reproduce

Use a dedicated **Python 3.11** environment, a C++ compiler and CPU PyTorch.
This setup was exercised on Windows with Visual Studio 2022 C++ tools and
PyTorch 2.13 CPU. The script includes Linux compiler flags; Linux compilation
has not been exercised here. JAX-MPM scenes use a separate Python 3.12+ environment.

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cutting.txt
python -m pip install '.[robot,test]'
python tools/setup_disect.py --destination ../disect-cpu
python -m softgriplab banana-cutting --backend ../disect-cpu --output outputs/banana_robot
```

Setup clones the pinned revision into a new external directory. It adapts
Python AST/import APIs, loads URDF/GUI support only if needed, locates the
Windows compiler and compiles the CPU kernels and C++ mesh cutter. It does
not edit the numerical kernels. `softgriplab_setup.json` records patch hashes.
An existing destination is never overwritten; `--resume-build` retries a
failed compilation after verifying those hashes. On Windows, an unusual
compiler location can be supplied with `SOFTGRIPLAB_VCVARS`.

Replay the committed result with only the normal robot dependencies:

```bash
python -m softgriplab animate-cutting --source docs/results/cutting/nominal
```

The renderer uses saved hard-node coordinates and interpolated virtual boundary
coordinates, with deformation scale 1. Recorded Panda joints reproduce the
knife contact mesh's center to within 1 micrometre. The mounting stem and end
fixture are visual markers; blade contact and the prescribed end constraint
are the physical model.

## Model

| Quantity | Value |
| --- | --- |
| Reference length / maximum diameter | 160 / 28 mm |
| Density / Young's modulus / Poisson ratio | 1,000 kg/m³ / 50 kPa / 0.30 |
| Reference mass | 76.570459 g |
| Virtual tetrahedra / cohesive springs | 2,592 / 243 |
| Planned planes in local x | 50, 3.3333, −43.3333 mm |
| Cohesive stiffness / damping | 500 N/m / 0.003 N·s/m |
| Knife contact penalty / friction | 1,000 N/m / 0.5 |
| Ground contact penalty / friction | 10,000 N/m / 0.6 |
| End constraint | Reference x < −72 mm; 50 fixed nodes |
| Robot coupling step | 1/120 s |
| Material step | 19.9840 µs; 417 substeps per coupling step |
| Duration | 4.2 s |

The robot's measured knife poses drive each interval's FEM substeps. The
opposite mean knife force loads the robot during the following interval.
This is first-order partitioned, force-only coupling: no contact torque is
returned. The blade is the same mesh in contact and rendering.

The three interfaces are inserted before the run, but start connected by
springs. The upstream law reduces spring stiffness only when knife contact
and positive spring opening coexist. Its per-step softening coefficient is
scaled by the material time step, using a rate of 2.5e7 N/(m²·s). This is an
illustrative opening-dependent damage law, not a calibrated fracture energy.

Preparing the virtual topology naively duplicates integration volumes: raw
lumped mass would increase from 76.57 to 98.22 g. `cut_quadrature.py` clips each
crossing reference tetrahedron at its cut plane, integrates its linear shape
functions over its own physical portion, and scales element coefficients by
that volume fraction. The corrected mass is 76.57 g. Analytical clipped-tet
tests check both volume and first moments. All cuts in this scene are separated
enough that no element crosses two planes; the adapter rejects that case.

## Evidence and limits

The [archived data](results/cutting) contains the full nominal node history,
topology, spring stiffness, measured robot joints, contact forces, settings,
backend source hashes and rendering provenance.

| Check | Recorded result |
| --- | --- |
| Complete nominal sequence | Cuts finish at 0.7417, 2.1250 and 3.5667 s |
| Material graph | 1 → 2 → 3 → 4 connected components; recomputed independently in tests |
| Reference mass preservation | Relative tolerance 1e-10 |
| Smallest determinant of deformation gradient | 0.3858, sampled every coupling interval |
| Ground penetration at saved physical surface nodes | At most 0.0519 mm |
| Zero-damage contact prefix, first 0.5 s | Nonzero knife reaction; all springs remain intact |
| Half material step, first 1.1 s | First cut completes at 0.7500 s, one robot interval later |
| Fresh backend clone/build and public CLI, first 1.1 s | All 13 observable columns match the nominal prefix exactly on this machine |

The two shorter controls cover their stated prefixes, not the full three-cut
sequence. The half-step comparison is a sensitivity check, not convergence
proof. The maximum tool tracking error is 9.92 mm during motion; contact always
uses the measured pose. Ground penetration over all mathematical support nodes
is reported separately because ghost nodes can lie outside the physical domain.

This is an **uncalibrated fruit surrogate with planned fracture planes**. It has
no peel, fibres, viscoelasticity or automatic crack-path discovery. The upstream
paired-interface contact is retained, but general collision between arbitrarily
rotated fragments is absent. There is no material/robot-arm collision. Spatial
convergence, measured cutting forces and real-robot validation remain open.

To regenerate the comparison records and archive:

```bash
python -m softgriplab banana-cutting --backend ../disect-cpu --duration .5 --softening-rate 0 --no-render --output outputs/banana_control
python -m softgriplab banana-cutting --backend ../disect-cpu --duration 1.1 --dt .00001 --no-render --output outputs/banana_fine
python tools/collect_cutting.py
python -m pytest -q tests/test_cut_quadrature.py tests/test_cutting_archive.py
```
