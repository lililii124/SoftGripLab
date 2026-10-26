# SoftGripLab

用接触与材料力学研究机器人抓取、黏附和切割。

[English](README.md) · [复现动画](docs/ANIMATIONS.md) · [模型](docs/MODELS.md) · [验证](docs/VALIDATION.md)

### 压缩与回弹

![压缩与回弹](docs/assets/01_compression.gif)

### 黏附接触

![压入与拉脱](docs/assets/02_adhesion.gif)

左：无黏附；右：有黏附。

### 柔性物体抓取

![夹持、提起、释放](docs/assets/03_grasp.gif)

### 机器人抓取搬运

![Panda 抓取刚性方块](docs/assets/04_robot.gif)

### 黏附搬运

![Panda 搬运柔性物体](docs/assets/05_adhesive_robot.gif)

当前展示[关闭黏附后释放](docs/ADHESIVE_ROBOT_SWITCH.md)；加速度上提脱粘正在验证。

### 连续切割

![Panda 连续切割弯曲果体](docs/assets/06_banana_cutting.gif)

[切割模型与复现](docs/CUTTING.md)。

动画来自求解后的材料状态与机器人位姿。小图为同步记录的轴向力随时间变化。

### 运行

```bash
python -m pip install '.[robot,test]'
python -m softgriplab fem-demo
python -m softgriplab robot-demo
python -m pytest -q
```

[黏附](docs/ADHESIVE_ROBOT.md) · [切割](docs/CUTTING.md) · [完整命令](docs/ANIMATIONS.md)

### 材料参数辨识

用 SciPy 从力—位移历史中辨识松弛参数：

```bash
python -m softgriplab demo --output outputs/demo
```

辨识示例采用合成数据；果体材料尚未经过试验标定。
复用 Bullet FEM、JAX-MPM、PeriPy 和 Panda 模型。[第三方说明](NOTICE) · [GPL-3.0](LICENSE)。
