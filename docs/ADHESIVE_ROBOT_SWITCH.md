# Adhesive robot / deformable payload

The Panda presses, engages adhesion, lifts, carries, sets down and releases a
30 mm elastic block with mass 27 g. The animation replays MPM material positions
and measured PyBullet joints at deformation scale 1:1. Rendering checks that the
visible tool center agrees with the contact center to within 1 micrometre.

## Model

| Quantity | Value |
| --- | --- |
| Elastic model | Neo-Hookean, E = 12 kPa, Poisson ratio 0.3 |
| Density | 1,000 kg/m3 |
| Particles / grid | 4,096 / 32 cubed over a 180 mm domain |
| Head radius | 18 mm |
| Adhesion work / effective range | 6 J/m2 / 0.8 mm |
| Contact penalty | 48 MPa/m |
| Robot coupling step | 1/120 s |
| MPM substep | 49.31 microseconds, 169 per robot step |
| Floor | z = 30 mm, grid-wall friction coefficient 0.5 |

Each robot step uses the previous interval's average material reaction. Measured
start and end tool centers define a linear path over the MPM substeps. Upstream
Dugdale traction acts on the current positions of the original top material
sampling layer. Equal and opposite forces load the next robot step. No material
positions are prescribed or reset; the payload has gravity and no anchors.

Adhesion can engage only after compressive contact. It is enabled from 1.25 s
and disabled at 5.05 s; repulsive contact stays active. This ideal switchable
cohesive interface does not model vacuum, sealing, electrostatics or curing.
The 0.8 mm range is an effective model parameter, not a measured microscopic
adhesion length. Elastic and interface parameters are illustrative.

## Recorded checks

Contact starts at 1.167 s. The center of mass rises from 45.0 to 96.54 mm and is
released within approximately 1.1 mm of the target. During 2.8-3.8 s, every saved
coupling step has active adhesion and the whole material clears the floor. Mean
upward contact force is 0.26486 N, compared with a 0.26487 N weight. Final contact
force is zero. The zero-work control makes the same initial compressive contact
and stays near its starting position.

Minimum sampled det(F) is 0.763; maximum contact penetration is 0.473 mm. The CSV
allows free-flight momentum change to be checked against integrated head force
and gravity. Floor-contact intervals are excluded from that balance check.

Halving the coupling step to 1/240 s preserves success. Peak height differs by
less than 0.001 mm and final center position by less than 0.02 mm. This checks one
coupling step on one grid; it does not establish spatial convergence or real
robot performance. Only the head exchanges contact with MPM. The floor has grid
discretization error, and the elastic body does not inherit the separate SLS fit.

## Reproduce

Use Python 3.12+ and the pinned JAX-MPM dependency:

```bash
python -m pip install '.[mpm,robot]'
python -m softgriplab adhesive-robot --output outputs/adhesive_robot
python -m softgriplab adhesive-robot --work 0 --no-render --output outputs/adhesive_robot_control
python -m softgriplab adhesive-robot --coupling-dt 0.004166666666666667 --no-render --output outputs/adhesive_robot_fine
```

Set `JAX_PLATFORMS=cpu` to explicitly select CPU. Windows may need MSVC to build
PyBullet. Replay requires only the ordinary package plus `[robot]`:

```bash
python -m softgriplab animate-adhesive-robot --source docs/results/adhesive_robot/nominal
```

Committed trajectories contain the exact outer material coordinates used for
rendering. Full particle histories stay in solver output. The archive retains
the control, CSVs, source hashes and coupling-step comparison.
`tools/collect_adhesive_robot.py` checks the records before updating public data.
