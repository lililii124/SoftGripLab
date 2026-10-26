# SoftGripLab

从接触力学走向机器人操作：**先识别材料，再理解接触，最后完成抓取与搬运。**

首页按由简单到复杂的顺序展示四组动画。曲线和参数表保留在验证材料中，主要展示物体、夹具、接触状态和机器人动作。

## 1．按压与材料辨识

![快慢加载下的材料压缩与力松弛](docs/assets/01_compression.gif)

观察两种加载速率下的压缩和力松弛，再用 SciPy 联合拟合黏弹性参数。
动画几何是由合成记录驱动的均匀压缩示意，力值来自记录，不是三维有限元变形场。

## 2．表面黏附与拉脱

![Hertz 和 JKR 接触对照](docs/assets/02_adhesion.gif)

对照无黏附 Hertz 接触与 JKR 黏附接触：回撤时，JKR 接触仍能承受拉力，直到拉脱。
接触半径、接触力和附着历史直接调用现有 JAX-MPM 接触模型；表面外形是示意。
这里描述的是表面黏附，区别于下一阶段的真空吸盘。

## 3．可变形物体夹持与扰动

![无摩擦与有摩擦夹持的 MPM 轨迹对比](docs/assets/03_grasp.gif)

软块经历夹紧、重力加载和基座加速度脉冲。动画中的粒子位置和变形颜色来自真实的三维 MPM 计算，展示其中央截面。
无摩擦组会下落，有摩擦组受到夹爪支撑。网格接触仍存在离散误差，当前结果用于方法演示。

## 4．机器人吸附搬运

![Panda 机械臂吸附搬运及末端特写](docs/assets/04_robot.gif)

复用 PyBullet 自带 Panda 模型，完成接近、接触、建立吸附、提起、搬运、释放和退回。
机器人由逆运动学和关节电机驱动；物体没有在运行过程中被重置位置。
吸附需要满足接触、覆盖范围与方向条件，并受压差、吸附面积和声明的力矩限制约束。
吸附能力不足的对照会触发连接断开，无法提起物体。

**机器人案例当前使用刚性物体和简化吸附连接，尚未与前面的黏弹性／MPM 模型耦合。**
这一区分也显示在动画与验证记录中，避免把不同层次的模型误认为完整的柔性机器人仿真。

## 运行

基础环境 Python 3.11+：

```bash
python -m pip install '.[test]'
python -m softgriplab demo
python -m softgriplab animate-compression
```

机器人演示：

```bash
python -m pip install '.[robot]'
python -m softgriplab robot-demo
```

JKR 与 MPM 使用 Python 3.12+ 的现有 JAX-MPM 环境：

```bash
python -m pip install '.[mpm]'
python -m softgriplab adhesion
python -m softgriplab mpm-grasp --output outputs/mpm
python -m softgriplab mpm-grasp --friction 0 --output outputs/mpm_frictionless
python -m softgriplab animate-grasp
```

完整复现说明见 [动画说明](docs/ANIMATIONS.md)；模型范围与验证依据见 [模型说明](docs/MODELS.md)、[验证记录](docs/VALIDATION.md)。
