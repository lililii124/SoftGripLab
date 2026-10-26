# Experimental acceleration-driven pull-off

**Status:** a successful inertial release has not yet been validated. The gallery
and default command retain the [verified switch-release case](ADHESIVE_ROBOT_SWITCH.md).
The commands below are investigation cases, not completed demonstrations.

The experimental adapter commands the Panda to press a spherical head onto a 27 g elastic block, lift and carry it, then accelerate upward. Adhesion stays enabled throughout withdrawal.
JAX-MPM solves material deformation and contact; the renderer replays saved
material coordinates and measured robot joints at deformation scale 1:1.

## Mechanics

For a payload following an upward acceleration a, the required resultant support
is approximately m(g+a). In the deformable model, interface force and payload
acceleration are solved. Detachment occurs when the material surface leaves the
finite attraction range. Upward movement alone does not guarantee release.

The 40 mm withdrawal command has constant positive acceleration for half its
duration, followed by braking. For duration T, the command magnitude is 4H/T².
Motor dynamics and the material reaction affect the actual motion. Every coupling
step records head position, velocity and acceleration, together with payload
velocity, interface gap, active contact samples and reaction force.

| Quantity | Value |
| --- | --- |
| Bulk | Neo-Hookean; E = 12 kPa; nu = 0.3; density 1,000 kg/m³ |
| Payload | 30 mm cube; weight 0.26487 N; gravity on; no anchors |
| Head | Sphere, radius 18 mm |
| Interface | Dugdale dry adhesion; effective range 0.8 mm |
| Compression | Penalty stiffness 48 MPa/m; always active |
| Mesh | 40³ background grid; exact particle count in each summary |

Adhesion engages only after compressive contact. In inertial mode its parameters
remain fixed and its activation remains on. Leaving the attraction range gives
zero force through the contact law itself. Each fast/slow pair uses the same interface parameters from the start.
The 2 J/m² trial failed to lift; it is not evidence of inertial release.
The 3 J/m² dense-grid trials were rejected by the state validator.
The stronger 6 J/m², grid-32 trials remained attached during withdrawal. These are illustrative parameters, without experimental calibration.
The model does not include a vacuum seal, leakage or irreversible interface damage.

## Checks

A successful inertial-release demonstration must first carry the block clear of
the table, maintain adhesive activation, then open the interface beyond its
attraction range and reach zero head force. The matched weak-acceleration case
must remain attached. Halving the coupling step checks the release outcome.
During free flight, integrated interface force and gravity are compared with the
payload momentum change. Floor-contact intervals are excluded from that balance.

Robot motion is simulated, not a hardware-qualified trajectory. Only the head
exchanges contact with MPM, and the partitioned reaction has one interval of lag.

## Reproduce

Use Python 3.12+ and the pinned JAX-MPM dependency:

```bash
python -m pip install '.[mpm,robot]'
python -m softgriplab adhesive-robot --release-mode inertial --grid 40 --work 4 --dt-factor .5 --release-duration .04 --output outputs/adhesive_fast
python -m softgriplab adhesive-robot --release-mode inertial --grid 40 --work 4 --dt-factor .5 --release-duration 1.2 --output outputs/adhesive_slow
python -m softgriplab adhesive-robot --release-mode inertial --grid 40 --work 4 --dt-factor .5 --coupling-dt 0.004166666666666667 --no-render --output outputs/adhesive_fine
python -m softgriplab animate-adhesive-robot --source outputs/adhesive_fast
```

Set `JAX_PLATFORMS=cpu` for CPU execution. Replay needs PyBullet, without JAX.
The renderer checks the replayed head against the contact center to within one
micrometre. Extra solved frames near withdrawal provide slow-motion
detail without image text. Model settings and forces remain in JSON/CSV, keeping the animation concise.

The [earlier switch-off example](ADHESIVE_ROBOT_SWITCH.md) and its archived checks
describe a different release mechanism; they are not evidence for inertial pull-off.
