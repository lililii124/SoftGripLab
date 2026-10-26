# Hyperelastic and adhesive parameters on a complex shape

![Solved truth and identified material](results/shape_identification/shape_identification.gif)

Left: synthetic reference material. Right: the identified parameters, independently
resolved on the same dense mesh. Both clips replay actual equilibrium node
positions. Insets show the tool's axial reaction. The curved, lobed specimen is
clamped at its two ends; the sphere is a prescribed probe.

## Run

```bash
python -m pip install '.[shape,test]'
python -m softgriplab shape-identification
```

The first run builds an offline finite-element response library, then identifies
off-grid synthetic materials and solves a dense validation case. CPU execution
takes minutes. Cached library curves support resuming the calculation.

Reuse the archived response library:

```bash
python -m softgriplab shape-identification --library docs/results/shape_identification/library.npz
```

Skip the dense validation and animation with `--no-dense --no-render` for inverse
method comparisons. Replay the archived dense states without rerunning FEM:

```bash
python -m softgriplab animate-shape-identification --source docs/results/shape_identification
```

## Mechanics

The known reference geometry is a curved solid with varying cross-section and
necks. Its bulk uses compressible Neo-Hookean elasticity:

```text
psi(F) = mu/2 * (F:F - 3) - mu*log(J) + lambda/2 * log(J)^2
J = det(F),  lambda/mu = 2*nu/(1-2*nu)
```

This version identifies shear modulus `mu` and work of adhesion `W`. Poisson
ratio `nu = 0.35`, geometry, fixtures and interface length scales are known.
It does not simultaneously estimate geometry or a general hyperelastic law.

For partial scans with unknown reference geometry, see the separate
[surface reconstruction and material-fitting benchmark](UNKNOWN_SURFACE.md).

The sphere interacts with the deformed boundary using its actual signed gap.
Positive gap gives a finite-range, triangular tensile traction; negative gap
gives compressive penalty contact. The traction integral is `W`. Forces and the
consistent contact tangent derive from the same surface potential. Reference
surface areas are lumped onto boundary nodes. Adhesion remains enabled during
withdrawal; separation follows from the solved gap.

Volume quadrature, vector tetrahedral bases and sparse assembly come from
[scikit-fem](https://scikit-fem.readthedocs.io/en/stable/). The bulk follows the
[standard total-Lagrangian Neo-Hookean formulation](https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex43.py).
SciPy solves Newton increments. An energy line search rejects inverted elements
and non-descending steps. Tests independently differentiate total energy and
residuals, check rigid-rotation objectivity and verify fixture/tool force balance.

## The inverse method

For fixed `nu`, no gravity, and the stated dimensionless contact penalty,
nondimensionalization gives

```text
eta = W / (mu * L)
u = L * u_hat(eta)
F_tool = mu * L^2 * F_hat(eta)
```

Thus the nonlinear FEM library needs only one parameter axis, `eta`. Each solved
shape can be reused at any compatible modulus scale. The offline library probes
three spatially separated regions and contains complete loading/unloading paths.
This is an exact scaling of this model; interpolation between ratio samples is
an approximation and is evaluated with direct, off-grid FEM data.

At a trial `eta`, the force least-squares optimum in `mu` has a closed form.
Physical bounds on both `mu` and `W` are enforced. The profile objective is
scanned within each library interval to bracket distinct local minima; those
brackets are refined together with
[SciPy's elementwise minimizer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.elementwise.find_minimum.html).
Grid and boundary candidates are retained. A scalar interval-search ablation
checks numerical agreement. This is a numerical multibasin search, not a proof
of global optimality. The method removes the need to optimize two coupled
parameters simultaneously. It is a use of established
dimensional analysis and variable projection, not a claim that these ideas are new.

Comparisons preserve the same response library, observations and parameter
bounds:

- Profile fitting of force only.
- Four-start, bounded continuous least-squares fitting of the same force data.
- A finite candidate-grid reference, which exposes parameter quantization.
- Profile and multistart fitting with additional visible-surface displacement.
- The unbatched scalar profile as a numerical/efficiency ablation.

Surface observations are area-weighted displacements of three material patches,
not hidden internal stress or ground-truth constitutive parameters. Additional
channels are not assumed to help when interpolation/model errors exceed sensor
noise. Both successful and unfavorable comparisons are saved.

## Evidence and boundaries

The [report](results/shape_identification/report.json) and
[per-trial metrics](results/shape_identification/metrics.csv) contain actual errors
and online timings. Validation materials are sampled with a separate fixed seed
after the development cases; their clean responses come from direct FEM solves,
not from the interpolation used for inference. Repeated sensor-noise trials are
paired across methods and are **not independent materials**.

The current six-material validation, with 24 paired noise repeats per material:

| Method | Mean shear-modulus error | Mean adhesion-work error |
|---|---:|---:|
| Continuous profile, force | 1.57% | 3.34% |
| Four-start continuous fit, force | 1.62% | 3.31% |
| Discrete candidate reference, force | 6.67% | 7.90% |
| Continuous profile, force and surface | 2.55% | 3.77% |

The gain over the discrete reference is substantial, but accuracy is comparable
to the strong continuous baseline. Extra surface channels do not improve this
library's result. Whitened fit residuals are also reported: interpolation error
exceeds the assumed sensor noise in some traces, so these fits must not be
presented as calibrated probabilistic estimates.

Batching provides the online efficiency gain while preserving the eliminated
likelihood. Exact measured speed ratios and the maximum objective difference
against both scalar and multistart methods are recorded under `performance` in
the report. These timings use the same cached response library and must not be
interpreted as end-to-end FEM or robot speedups.

The dense comparison resolves the reference and identified parameters again on
a separate mesh. It checks one material's transfer; it is not a mesh-convergence
study. Rendering does not subdivide or interpolate a coarse displacement field
to fabricate a dense simulation.
The replay uses the scalar-profile ablation's identified parameters; batched
and scalar objective agreement is audited on every archived noise trial.

For that dense pair, the pull-off peak differs by about 0.75% and mean absolute
force discrepancy is 2.46 mN. The whole-curve relative L2 discrepancy remains
30.6%, including differences near contact jumps. The full curve therefore does
**not** constitute a successful high-accuracy transfer result, even though the
peak agrees. This limitation is retained in the report and animation.

The primary method benefit is eliminating one nonlinear parameter axis and
avoiding repeated full FEM solves during online identification. Online timing
must be read alongside the offline library cost. It does not by itself establish
superiority over strong continuous optimizers, a novel research contribution,
or an advantage from actively selecting probe locations.

## Small-data constraint

The current implementation has no neural-network training. It uses 13 material
ratio samples at three probe sites. Noise repeats are validation only. Future
PINN, energy-network or residual-learning extensions must use physical equations
and a small, explicitly reported calibration budget. They should be compared
with this baseline using the same observations and held-out materials; no large
synthetic training campaign is assumed or required.

This is a gravity-free, quasi-static, normal-contact reference. It excludes
friction, suction/sealing, dynamics, viscoelasticity and irreversible interfacial
damage. The interface has reversible attraction on recontact. Frame times index
a prescribed quasistatic loading schedule; they are not a validation of transient
release dynamics. Geometry, constitutive law and sensing are synthetic; no real
material or robot experiment is claimed.
