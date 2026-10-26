# Models and interpretation

## Uniaxial compression

Known initial thickness h and loaded area A, uniform small strain epsilon=d/h,
positive compression. The standard linear solid is a spring E_inf in parallel
with one Maxwell branch (E_b, tau). Its relaxation modulus is

    E(t) = E_inf + E_b exp(-t/tau).
    z' = (epsilon - z)/tau, z(0-) = 0.
    sigma = (E_inf + E_b) epsilon - E_b z.
    N = A sigma.

The state-space dynamics are integrated by SciPy `signal.lsim` with linear
interpolation of equally spaced samples. An initial nonzero input means an
instantaneous jump from zero prehistory. The inverse transfer function gives
ideal force-controlled creep. This is a uniaxial material response, not an
indentation geometry or unilateral contact solver. Lateral constraint, changing
contact area, large strain and fixture compliance are not modeled. Real
indentation data cannot be interpreted as uniform strain without a suitable
forward model.

Optimization uses positive log parameters, bounded SciPy `least_squares`, and
three deterministic initializations. Residuals are divided by each trace's
specified noise standard deviation. Condition number and local covariance use
the weighted log-parameter Jacobian. These diagnostics assume local linearity
and independent residuals; they do not quantify constitutive-model uncertainty.
A poorly excited record may not identify three separate parameters.

## Two-finger vertical grasp

N is the normal force on each finger. With symmetric contact and known Coulomb
coefficient mu, the total tangential capacity is 2 mu N. The required support
force for vertical frame acceleration a is m |g+a|. The reported margin is

    margin = 2 mu N - m |g+a|.
    design_margin = 2 mu N - safety_factor * m |g+a|.

The constraint on total closure is d/h <= strain_limit. Negative margin means
that sticking is infeasible in this reduced model. It does not predict actual
slip distance, a dropped object, stress concentrations, or damage. The contact
area and friction coefficient are assumed known and constant.

The planner scales the response to a unit strain ramp and picks the smallest
constant closure satisfying the supplied acceleration history after load
transfer. It returns infeasible demands explicitly. It knows the disturbance
envelope; there is no claim of adaptation to arbitrary unseen impulses.
Parameter confidence bounds are reported by fitting but not yet propagated into
the planner. The force-control comparison assumes unlimited controller bandwidth.

For the half-sine pulse a=A sin(pi (t-t0)/T), the equivalent inertial impulse is
2 m A T/pi. This is base excitation, not an impact/contact calculation.

## 3D integration

The optional scene calls `mpm_jax.step` without altering or reimplementing it.
JAX-MPM handles particle/grid transfers, Neo-Hookean stress, explicit integration,
plane-wall Coulomb contact and validity checks. This scene uses its elastic
material, not the standard linear solid above. A `young-pa` value supplied from a
fit can only be an explicitly chosen elastic approximation, not transfer of the
identified rate dependence.

Two infinite planar jaws move symmetrically for 0.03 s and then hold. Gravity
loads the object progressively over the next 0.015 s. A 15 ms half-sine vertical
frame-acceleration pulse starts at 0.075 s. External loads are distributed by
particle mass. The cube has 30 mm edges and density 1000 kg/m³. The reference
benchmark uses a different specimen and payload; results are not interchangeable.

The upstream grid boundary geometry has grid-scale error, especially when the
closure is smaller than a grid cell. Time-step checks cannot substitute for
spatial convergence. Green-strain norm is a deformation diagnostic, not a damage
indicator. `validate_state` runs at every saved frame; upstream state validity
accumulates every integration step. The exported minimum determinant is sampled
at frames and is labeled accordingly.

## Volumetric FEM scenes

Bullet's deformable world integrates a compressible Neo-Hookean solid on a
structured tetrahedral mesh. Young's modulus is 15 kPa, Poisson ratio 0.35,
density 1000 kg/m³ and block dimensions 50 × 50 × 40 mm. The default mesh has
2,028 nodes and 9,504 positive-volume tetrahedra. Lamé constants are passed to the
upstream solver. The numerical damping setting is 0.03 and contact margin 0.3 mm.

Compression prescribes a finite upper platen's displacement and then unloads it.
The grasp uses two finite rectangular jaws, gravity and Coulomb friction, followed
by jaw opening. Both fixture positions and velocities are prescribed. The latter
is essential to transmit tangential friction during lifting. Specimen nodes have
no anchors. The deformation gradient of each tetrahedron is computed from its
current and reference edge matrices; its determinant is checked at every step.
Surface rendering reads the saved nodes and topology without shape interpolation.

## Continuum adhesive contact

The main adhesion scene uses the pinned upstream `DugdaleLaw`, `SpherePath` and
`sphere_interaction`. On the current top material sampling layer, a negative gap
gives penalty repulsion, a gap between zero and the finite interaction range gives
attractive traction, and a gap outside the range gives no traction. Forces are
integrated using reference surface areas and supplied to `mpm_jax.step`.
The bulk response changes the gaps at the next step, closing the mechanics loop.

The matched comparison uses E=8 kPa, nu=0.25, a 16 mm sphere, work of adhesion
0 or 14.4 J/m², interaction range 2.5 mm and penalty stiffness 48 MPa/m. The base
is fixed, gravity is zero, and the sphere follows a prescribed indentation,
withdrawal, hold and final lift trajectory. Tangential adhesive traction is absent.
This finite-range dry-interface model is not a vacuum seal or calibrated material.
`adhesion` retains the independent analytical Hertz/JKR cycle as CSV only.

## Articulated robot contact

The Panda scene closes two real finger joints on a 50 mm, 50 g rigid cube. The
arm and fingers use position motors; a 4 N motor limit is set for each finger.
The object friction coefficient is 0.8 and finger coefficient is 1.0. No payload
attachment joint or collision exclusion is created. The demo records both normal
forces and signed contact distances at every step, and checks continuous bilateral
contact during lift/transfer. The zero-friction control fails to lift.

The robot payload is rigid. This scene and the FEM grasp have different objects,
controllers and contact discretizations; they are not a coupled soft-robot model.
The reduced `SuctionSpec` capacity utility remains available for unit calculations,
but it is no longer used to animate a robot attachment.

## Adhesive Panda with a deformable payload

Scene 5 couples motor-driven Panda dynamics to an unanchored Neo-Hookean MPM
block. The measured sphere center drives upstream Dugdale contact on the actual
top material layer. Its interval-averaged reaction loads the next robot step.
Gravity acts throughout; switching adhesion off after landing releases the
object. A compression-contact gate prevents engagement before pressing.
Parameters, controls, time-step comparison and limitations are recorded in
[the published switch-release scene](ADHESIVE_ROBOT_SWITCH.md).
The acceleration-driven release adapter is experimental; see [its checks](ADHESIVE_ROBOT.md).

## References

- [SciPy lsim](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lsim.html)
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
- [Constitutive equations for finite-rate SLS loading, Polymers 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9143375/)
- [JAX-MPM documentation](https://github.com/lililii124/jax-mpm/blob/main/docs/EXAMPLES.md)

## Sequential cutting

[Scene 6](CUTTING.md) uses unmodified PeriPy microelastic force and integration
kernels with a moving finite-blade adapter. The initial material is intact and
settles onto the board under gravity. Bonds crossing the measured blade surface
are removed during integration; no cut surfaces are inserted before the run.
The robot receives the material reaction force and torque. The renderer reads
solved particles and reconstructs each connected component without the knife path.
The fruit surrogate is homogeneous and uncalibrated; the geometric cutting rule
does not predict fracture-energy-dependent cutting forces. The previous planned
DiSECt example remains [archived](CUTTING_DISECT.md).
