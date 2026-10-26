# Roadmap

## v0.1 — executable reference and continuum integration

- [x] Multi-rate material identification, external CSV input and fit diagnostics.
- [x] Held-out loading and constitutive-mismatch probes.
- [x] Three grasp baselines and disturbance/strain feasibility envelope.
- [x] Existing MPM backend integration, planar jaws and base acceleration.
- [x] Analytic and inference tests; optional continuum momentum check.

## v0.3 — solver geometry and observable contact

- [x] Volumetric Neo-Hookean FEM compression, recovery and finite-jaw grasp.
- [x] Saved solver meshes rendered at 1:1 scale with correct depth occlusion.
- [x] Upstream Dugdale tractions coupled to actual MPM surface deformation.
- [x] Panda bilateral frictional grasp, joint motors and contact close-up.
- [x] Zero-friction / zero-adhesion controls and replayable surface data.
- [x] Neutral engineering colors; synthetic geometry removed from the main gallery.

## Next — one material model across identification and grasping

- [ ] Reuse a supported upstream viscoelastic law or contribute it upstream;
      do not fork a complete material/particle solver into this project.
- [ ] Independent sliding trials for friction, multiple normal loads.
- [ ] Separate per-jaw force observations and geometry-aware calibration.
- [ ] Spatial convergence, time-step convergence and transfer validation.
- [ ] Measured compression/sliding records with provenance and redistribution rights.
- [ ] Propagate material/friction uncertainty into grasp decisions.

## Next — articulated deformable grasp and protective pads

- [ ] Extend the existing PyBullet Panda demo to deformable pads and finite contact.
- [ ] Reuse existing deformable contact for finger pads and localized impactors.
- [ ] Measure slip trajectory, peak contact force, energy dissipation and recovery.
- [ ] Calibrate rate effects using multiple loading rates before impact predictions.
- [ ] Add damage only with a documented constitutive law and calibration evidence.

Learning-based policies come after reliable force/geometry observations and
baseline validation. They are not required to demonstrate the mechanics contribution.
