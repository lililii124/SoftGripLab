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

## Adhesion and robot suction

The v0.2 contact gallery delegates Hertz/JKR evaluation and attachment history
to `mpm_jax.contact_laws`. It uses work of adhesion in J/m² and reduced modulus
E*, with separate load-controlled force minimum and displacement-controlled
detachment criteria. The renderer does not solve a surface displacement field.

The robot example uses PyBullet's Panda model and a pressure-capacity-limited
fixed constraint, acquired only after contact and seal-geometry checks. The
payload is rigid. Force and moment limits, the one-step trial-constraint
overshoot, and the distinction from dry adhesion are documented in
[ANIMATIONS.md](ANIMATIONS.md). These scenes are separate model levels, not a
coupled material-to-robot solver.

## References

- [SciPy lsim](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lsim.html)
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
- [Constitutive equations for finite-rate SLS loading, Polymers 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9143375/)
- [JAX-MPM documentation](https://github.com/lililii124/jax-mpm/blob/main/docs/EXAMPLES.md)
