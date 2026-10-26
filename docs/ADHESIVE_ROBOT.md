# Acceleration-driven adhesive release

The Panda presses onto an elastic block, lifts and carries it, then accelerates
upward. The block detaches with adhesion still enabled and lands on the table.
The animation replays solved JAX-MPM material coordinates and measured PyBullet
joints at deformation scale 1:1. Withdrawal is shown at 8x slow motion.

## Mechanics

For a payload following an upward acceleration a, the required resultant support
is approximately m(g+a). In the deformable model, interface force and payload
acceleration are solved. Detachment occurs when the material surface leaves the
finite attraction range. Upward movement alone does not guarantee release.

The head's 40 mm withdrawal command accelerates for half its duration, then
brakes. Motor dynamics and material reaction determine the actual motion. A
0.04 s command gives a measured peak upward acceleration of 37.14 m/s²; the
commanded value is 100 m/s². The slower 1.2 s command peaks at 0.112 m/s².

| Quantity | Value |
| --- | --- |
| Bulk | Neo-Hookean; E = 12 kPa; nu = 0.3; density 1,000 kg/m³ |
| Payload | 30 mm cube; weight 0.26487 N; gravity on; no anchors |
| Head | Sphere, radius 18 mm |
| Interface | Dugdale dry adhesion; work 4 J/m²; effective range 0.8 mm |
| Compression | Penalty stiffness 48 MPa/m; always active |
| Robot coupling step | 1/120 s; refined control 1/240 s |

Adhesion engages after compressive contact. Its activation and parameters remain
fixed during withdrawal; separation makes the contact law return zero force.
The solver does not switch adhesion off, reset material velocities or attach
the payload to the head.

## Matched checks

The [archived runs](results/inertial_release/comparison.json) share the same
material and interface parameters. Every recorded fast/slow state is identical
through 4.4 s, before the withdrawal histories diverge.

| Withdrawal | First separated sample | Outcome |
| --- | --- | --- |
| Fast | 4.4167 s | Detaches during positive upward acceleration; lands |
| Slow | None | Remains attached |
| Fast, half coupling step | 4.4208 s | Detaches during positive upward acceleration; lands |

Tests verify carriage clear of the table, continuous adhesive activation,
opening beyond the attraction range, zero force after separation, positive
material determinants and free-flight momentum balance. Force is averaged over
the preceding coupling interval: the first separated endpoint can still report
an interval force, followed by zero in fully detached intervals.

These are numerical consistency checks with illustrative material parameters.
Only the head exchanges contact with MPM; reaction feedback has one interval of
lag and floor contact has grid discretization error. Temporal agreement here
does not establish spatial convergence, experimental calibration or hardware
performance. The dry-adhesion model includes no vacuum seal, leakage or
irreversible interface damage.

## Reproduce

Use Python 3.12+ and the pinned JAX-MPM dependency:

```bash
python -m pip install '.[mpm,robot]'
python -m softgriplab adhesive-robot --output outputs/inertial_w4_fast
python -m softgriplab adhesive-robot --release-duration 1.2 --no-render --output outputs/inertial_w4_slow
python -m softgriplab adhesive-robot --coupling-dt 0.004166666666666667 --no-render --output outputs/inertial_w4_fine
python tools/collect_inertial_release.py
```

Set `JAX_PLATFORMS=cpu` for CPU execution. Replay needs PyBullet without JAX:

```bash
python -m softgriplab animate-adhesive-robot --source docs/results/inertial_release/fast
```

The renderer checks the replayed head against the contact center to within one
micrometre. Extra solved frames near withdrawal provide slow-motion detail.
Full-rate forces remain in CSV; surface positions and robot joints remain in NPZ.

The [earlier switch-release archive](ADHESIVE_ROBOT_SWITCH.md) is retained with
its own explicit reproduction parameters.
