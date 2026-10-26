# Validation record — 2026-10-26

This release provides numerical verification and integration examples. There are
no physical experiments or real-robot success rates in this record.

## Material identification and reference grasp

Run `python -m softgriplab demo --output outputs/demo` with seed 7.
Inputs are two analytic ramp/hold traces with independent Gaussian force noise
(standard deviation 0.005 N), durations 4 s, ramp times 0.12 and 0.8 s.
The numerical forward model uses SciPy `lsim`; it is not used to generate truth.

| Parameter | Truth | Fitted | Relative error |
| --- | ---: | ---: | ---: |
| Equilibrium modulus | 35000 Pa | 34990.796 Pa | 0.0263% |
| Maxwell branch modulus | 65000 Pa | 64861.418 Pa | 0.2132% |
| Relaxation time | 0.6 s | 0.600197 s | 0.0328% |

The unseen 0.035 s ramp has relative force L2 error **0.0917%**. This confirms
recovery for the assumed model and synthetic loading; it does not validate real
rate-dependent material behavior. Against a deliberately different two-branch
material, error rises to **11.23%**. That probe is reported without refitting.
The weighted Jacobian condition number is 7.32.

| Strategy | Minimum physical force margin | Peak compression strain | Meets configured design constraints? |
| --- | ---: | ---: | --- |
| Fixed closure | -0.0643 N | 4.000% | No |
| Ideal force control | +0.1595 N | 5.332% | Yes |
| Identified fixed closure | +0.1283 N | 5.337% | Yes |

Physical margin uses m|g+a|; the planner uses an additional factor of 1.2.
The strain cap is 8%. The identified plan is evaluated using the true synthetic
material, not the fitted model. Its closure is 1.0674 mm. Ideal force control
also succeeds; no universal superiority claim is made.

Machine-readable inputs, policies and results are in [results/](results/).

## Three-dimensional backend

Public JAX-MPM reference commit:
`fd8ed2caa40a50d01992420660f68ae6ce0d084f`.
The source files needed by the exercised API were retrieved/compared against
that commit in an isolated verification snapshot. Their Git blob hashes are in
[backend_api.json](results/backend_api.json). No upstream source is committed
into SoftGripLab. The existing local solver checkout was not modified.

Executed on CPU using Python 3.12.13 and JAX `0.12.1.dev20261022`.
Fresh optional-dependency installation and hosted CI were not exercised locally;
the source/API compatibility was exercised with the existing JAX runtime.

```bash
python -m softgriplab mpm-grasp --grid 24 --frames 40 --gif --output outputs/mpm
python -m softgriplab mpm-grasp --grid 24 --frames 40 --friction 0 --output outputs/mpm_frictionless
python -m softgriplab mpm-grasp --grid 24 --frames 40 --friction 0 --dt-factor 0.5 --output outputs/mpm_frictionless_halfdt
python -m softgriplab mpm-grasp --grid 32 --frames 40 --output outputs/mpm_grid32
python tools/collect_validation.py
```

| Check | Result |
| --- | --- |
| Grid 24, friction 0.6 | 1728 particles; 1120 steps; frame-sampled min det(F) = 0.88347 |
| Grid 24, frictionless | Final COM drop 40.638278 mm |
| Independent free-motion integral | Final COM drop 40.638597 mm |
| Time-step reduction | Absolute drop error decreases from 0.3193 µm to 0.0828 µm |
| Frictional peak drop, grid 24 | 0.45315 mm |
| Frictional peak drop, grid 32 | 0.48202 mm |

The free-motion oracle integrates `(end_time-t) * acceleration(t)` independently
with SciPy quadrature. With zero jaw friction, horizontal squeeze must not alter
vertical center-of-mass momentum. Its error reduces with the smaller time step.
This check validates that part of the loading/integration interface, not all
contact or constitutive behavior.

The two frictional grid resolutions differ by about 6.4% in peak COM drop.
**Spatial convergence is not established.** The imposed jaw closure is smaller
than a grid cell, so geometric contact error matters. These coarse scenes
demonstrate backend integration and contact response; they are not calibrated
predictions of a physical gripper's retention limit.

[Supplementary MPM comparison plot](assets/mpm_comparison.png). The original particle-section comparison remains available through `animate-grasp`;
the main grasp animation now uses finite FEM jaws.

## v0.1 automated checks (historical)

- Base Python 3.11 environment: 15 reference/CLI tests pass; the optional MPM
  test is skipped when JAX-MPM is absent.
- Python 3.12 with the pinned backend API: **16 tests passed**.
- Coverage includes analytic finite ramps, step relaxation, ideal creep,
  multirate noisy parameter recovery, held-out loading, short-record warnings,
  malformed input, two-contact force accounting, pulse impulse, infeasible plans,
  excessive creep, CSV CLI input and optional continuum momentum balance.

GitHub Actions definitions are included. Local success is not a claim that a
remote workflow has run or passed; consult the repository's Actions page.

The v0.1.0 wheel was built and installed into an isolated target directory.
The installed package's CLI reproduced the material fit from the committed CSVs.

## v0.3 solver geometry and contact verification

The main gallery now uses resolved deformable geometry. The earlier uniform
compression sketch, JKR surface sketch and suction-joint robot demonstration
are superseded. The independent Hertz/JKR numerical references remain tested.

| Case | Locally observed result |
| --- | --- |
| FEM compression, 1,764 tetrahedra | Height decreases from 40 mm to 30.407 mm; maximum width 54.483 mm |
| FEM unloading | Final height 39.393 mm with gravity still applied |
| FEM validity | Every-step minimum det(F): 0.7303 (compression), 0.6015 (grasp) |
| FEM grasp | Maximum center height 50.976 mm, starting at 20.500 mm; no node anchors |
| FEM zero friction, 600 tetrahedra | Maximum center height 21.005 mm; no meaningful lift |
| MPM adhesion, 14,700 particles | Maximum surface uplift 11.195 mm; tensile sphere reaction reaches -2.291 N |
| Same MPM path, no adhesion | Uplift at most 0.607 mm; no tensile reaction |
| MPM cycle completion | Both cases end detached; all upstream validity checks pass |
| Panda nominal | 100% bilateral contact during the checked airborne interval; no payload joint |
| Panda contact geometry | Maximum penetration 0.182 mm in the nominal run |
| Panda placement | Final center approximately (0.50004, 0.19373, 0.12499) m; target center (0.5, 0.18) m |
| Panda zero friction | Center never rises above 0.12501 m; pickup fails |

The FEM contact margin is 0.3 mm. Comparing 600 with 1,764 tetrahedra changes
maximum compression width by 0.146 mm (0.27%) and minimum height by 0.006 mm.
Halving the time step at 1,764 tetrahedra changes maximum width by 0.007 mm and
minimum height by 0.004 mm. Extrema are frame sampled, with 31 control frames
versus 91 gallery frames; these checks do **not** establish asymptotic spatial
convergence. Grasp/contact strain fields need further refinement.

The MPM interface range is 2.5 mm and maximum penalty penetration in the adhesive
case is 0.848 mm. Its local penalty step number is 0.0875. Deformation and traction
respond to one another, but the contact is coarse and the parameters are
illustrative. Pull-off loads are not presented as converged or calibrated values.

Local automated checks for this revision:

- Python 3.11 / PyBullet: **25 passed, 3 optional MPM modules skipped**.
- Python 3.12 / pinned MPM API: **4 selected backend tests passed**.
- Checks include mesh volume and outward topology, compression bulging and
  recovery, positive tetrahedral determinants, friction-dependent soft lift,
  bilateral robot contact, zero-friction failure, and zero-adhesion/tensile controls.

The renderer consumes saved coordinates at scale 1:1. FEM input hashes and the
matched MPM trajectory hashes are checked by the gallery collector. Compact
surface replay files retain the actual rendered float64 coordinates, topology,
fixture/sphere poses and time samples. They omit unrendered interior particles.
See [results/animations/](results/animations/) for CSVs, settings and replay NPZs.

The software stack uses Bullet's existing FEM and articulated-body solvers,
JAX-MPM's existing continuum/contact code, and the upstream Panda asset. No
experimental material calibration or real-robot validation is claimed.

## Acceleration-driven adhesive release

The [fast and slow withdrawal runs](results/inertial_release/comparison.json)
have identical recorded states through 4.4 s and identical material/interface
parameters. Both first carry the block clear of the table. Adhesion remains
enabled throughout release, with no anchors or payload attachment constraints.

The fast run first separates at 4.4167 s while the measured head acceleration is
34.33 m/s² upward. The slow run remains attached. Halving the robot coupling step
gives separation at 4.4208 s with positive upward acceleration. Both fast cases
subsequently have zero head reaction and land on the table. This checks the
release outcome under temporal refinement, without asserting spatial convergence.

Archive tests check active contact, gap, positive material determinants,
free-flight momentum balance, unchanged adhesion activation and render provenance.
The reported force averages the preceding coupling interval: its first separated
endpoint can still report force; subsequent detached intervals report zero.

## Legacy sequential cutting archive

[The archived DiSECt validation](CUTTING_DISECT.md#evidence-and-limits) records a complete
three-cut rollout, a 0.5 s no-damage contact prefix and a 1.1 s half-step first-cut
check. Tests independently reconstruct material connectivity, verify irreversible
spring failure, positive element determinants, reference mass, fixed nodes and
rendering provenance. These are numerical consistency checks, not experimental
validation or spatial convergence evidence.

## Dynamic cutting without initial interfaces

[The current cutting checks](CUTTING.md#archived-validation) use the saved
PeriPy material graph. Tests independently check intact initial connectivity,
support contact, irreversible bond removal, retained mass and four major final
pieces. A shifted stroke uses the identical initial graph and produces a different
cut. Disabling both severing and stretch failure preserves a contacted intact body.
Both tested time steps complete the three cuts, but fragment mass and contact
impulse remain sensitive to the time step; no converged force prediction is claimed.
Render hashes tie the animation and its force inset to the saved observations.
CSV checksums normalize CRLF to LF so the same record verifies on Windows and Unix.
