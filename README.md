# SoftGripLab

Contact and material mechanics for robotic grasping, adhesion, and cutting.

[中文](README.zh-CN.md) · [Run the demos](docs/ANIMATIONS.md) · [Models](docs/MODELS.md) · [Validation](docs/VALIDATION.md)

### Compression

![Compression and recovery](docs/assets/01_compression.gif)

### Adhesive contact

![Indentation and pull-off](docs/assets/02_adhesion.gif)

Without adhesion (left), with adhesion (right).

### Deformable grasping

![Grip, lift and release](docs/assets/03_grasp.gif)

### Robot pick and place

![Panda gripping a rigid block](docs/assets/04_robot.gif)

### Adhesive transport

![Panda carrying a deformable object](docs/assets/05_adhesive_robot.gif)

[Accelerate upward to detach](docs/ADHESIVE_ROBOT.md). Release shown in slow motion.

### Continuous cutting

![Panda cutting a curved fruit surrogate](docs/assets/06_banana_cutting.gif)

[Cutting model and reproduction](docs/CUTTING.md).

Animations replay solved material states and robot poses. Insets show recorded axial force over time.

### Run

```bash
python -m pip install '.[robot,test]'
python -m softgriplab fem-demo
python -m softgriplab robot-demo
python -m pytest -q
```

[Adhesion](docs/ADHESIVE_ROBOT.md) · [Cutting](docs/CUTTING.md) · [All commands](docs/ANIMATIONS.md)

### Material identification

**Complex-shape hyperelasticity and adhesion:**
[model, inverse method and reproduction](docs/SHAPE_IDENTIFICATION.md).

![3D reference and identified material](docs/results/shape_identification/shape_identification.gif)

Left: reference. Right: identified parameters. Both are solved again on a dense mesh.

**Active contact identification:** probe an unknown adhesive specimen and
predict release. [Run and inspect the benchmark](docs/ACTIVE_IDENTIFICATION.md).

![Two plausible materials, different release predictions](docs/results/active_identification/ambiguity_release.gif)

Axial reference model; synthetic observations. Policy comparisons are reported without assuming an active-policy advantage.

Fit relaxation parameters from force–displacement histories with SciPy:

```bash
python -m softgriplab demo --output outputs/demo
```

The bundled identification data are synthetic; the fruit model is uncalibrated.
Built on Bullet FEM, JAX-MPM, PeriPy and the Panda model. [Attribution](NOTICE) · [GPL-3.0](LICENSE).
