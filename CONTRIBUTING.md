# Contributing

Use a separate branch and include a runnable case for changes to physical models.
Keep SI units and the per-finger force convention explicit. New backends should
call the original solver and report the version/source revision used, rather than
copying solver implementations into this repository.

Run `python -m pytest -q`. If JAX-MPM is installed, the continuum integration test
runs as well. Update the validation record only after executing the corresponding
commands. Keep generated large trajectories under `outputs/`; commit only small
evidence files needed to understand reported results.

Label synthetic data, physical experiments and simulator-generated data
separately. Do not infer friction from normal compression alone, report
deformation as damage, or call a fitted synthetic trace experimental validation.

For new control or learning methods, keep the fixed-closure and ideal-force
baselines and declare their information and actuator assumptions.
