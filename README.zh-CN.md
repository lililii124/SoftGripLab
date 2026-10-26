# SoftGripLab

面向机器人接触、软物体夹持、材料参数辨识和扰动响应的力学基准库。
复用 **Bullet 有限元、JAX-MPM、SciPy 和 Franka Panda 模型**。

[English](README.md) · [复现动画](docs/ANIMATIONS.md) · [模型说明](docs/MODELS.md) · [验证记录](docs/VALIDATION.md)

## 1 · 试样压缩与回弹

![实体有限元试样的压缩、保压和卸载](docs/assets/01_compression.gif)

50 × 50 × 40 mm 的 Neo-Hookean 试样由 **9,504 个四面体单元**构成。
上下压板运动后，求解器计算压缩、侧向鼓胀和卸载回弹。渲染直接读取保存的
节点坐标，右侧剖面跟踪同一批材料节点；**变形倍率为 1**。这里采用弹性材料，
与后面的黏弹性参数辨识算例分开。

## 2 · 接触、黏附与拉脱

![有无黏附时的真实 MPM 表面变形](docs/assets/02_adhesion.gif)

两组软层采用相同的球形压头轨迹。左侧无黏附，右侧复用上游 Dugdale 有限作用距离
界面模型，根据**当前变形表面**计算作用力，并送入三维 MPM 求解器。
压头回撤时可观察到表面隆起和最终脱离。该模型描述干黏附，尚不描述真空吸盘的
密封、气压演化和泄漏。

## 3 · 软块夹持、抬升与释放

![有限尺寸夹爪接触并抬起有限元软块](docs/assets/03_grasp.gif)

两个有限尺寸夹爪先夹紧，再通过摩擦抬起软块，张开后让软块落回支承面。
软块节点没有绑定到夹爪。无摩擦对照无法抬起试样。
夹爪位置、表面变形和剖面节点均来自保存的仿真状态。

## 4 · Panda 双指搬运

![机器人真实双指接触、搬运和接触近景](docs/assets/04_robot.gif)

关节电机驱动双指夹住**刚性**方块并搬运到目标位置。右侧近景显示指尖接触，
两侧法向力和带符号的接触距离保存在 CSV 中。没有物体绑定关节，也没有关闭指尖碰撞。
摩擦系数置零后无法提起物体。该机器人场景与前面的软体夹持场景分别求解，
目前没有把刚性方块包装成可变形物体。

## 5 · 黏附抓取可变形物体

![Panda 黏附承重、搬运和释放，近景回放实际材料网格](docs/assets/05_adhesive_robot.gif)

Panda 先压触 **27 g 的软块**，再开启黏附、抬升、横向搬运 80 mm，落台后关闭黏附。
机械臂实际末端位置驱动 MPM 接触计算，材料反力在下一耦合步反馈给机械臂。
软块承受重力，没有固定节点或物体绑定关节。关闭黏附的对照留在原位。

右侧近景直接回放保存的材料网格与机械臂关节，变形倍率为 1。
[模型、验证与回放数据](docs/ADHESIVE_ROBOT.md)。

## 6 · Panda 连续三次切割

![Panda 切割香蕉形软体，材料依次分离为四块](docs/assets/06_banana_cutting.gif)

机械臂连续完成三次进刀、下切和回撤。**有限元网格受刀具接触后变形**，内聚弹簧损伤断开，
材料连通块从 1 个依次变成 4 个。每次切割延续前一次的材料状态，刀具反力反馈给机械臂。
近景直接回放求解器节点，物体左端采用固定边界。

该算例复用 **DiSECt 可选研究后端**，遵循其单独的 NVIDIA 研究许可。
目前使用未标定的均匀香蕉形材料和三条预设断裂面。[模型、对照、许可与复现](docs/CUTTING.md)。

## 运行

Python 3.11+，CPU 即可：

```bash
python -m pip install '.[robot,test]'
python -m softgriplab fem-demo --output outputs/fem_compression
python -m softgriplab fem-demo --scene grasp --output outputs/fem_grasp
python -m softgriplab robot-demo --output outputs/robot
python -m softgriplab robot-demo --friction 0 --no-render --output outputs/robot_frictionless
python -m pytest -q
```

黏附场景需要 Python 3.12+ 和固定版本的 JAX-MPM：

```bash
python -m pip install '.[mpm,robot]'
python -m softgriplab mpm-adhesion --output outputs/continuum_adhesion
python -m softgriplab mpm-adhesion --work 0 --output outputs/continuum_nonadhesive
python -m softgriplab animate-adhesion
python -m softgriplab adhesive-robot --output outputs/adhesive_robot
python -m softgriplab adhesive-robot --work 0 --no-render --output outputs/adhesive_robot_control
```

求解与渲染可以放在不同环境中。`animate-adhesion` 读取 NPZ 数据，只需要 PyBullet，
不导入 JAX。有限元求解支持 `--no-render`，随后用
`animate-compression --source outputs/fem_compression` 单独渲染保存的结果。

每组场景输出动画、预览图、设置和观测数据。软体场景保留节点或粒子坐标、网格拓扑；
有限元还输出初始 OBJ 表面和四面体 VTK 网格。[完整复现命令](docs/ANIMATIONS.md)。

## 材料参数辨识与抗扰动基准

`demo` 使用 SciPy 拟合标准线性固体的多速率压缩记录，检验未参与拟合的加载历史，
并比较夹持力基线。随库数据为合成数据。自己的试验记录需使用 SI 单位，CSV 列名为
`time_s,compression_m,force_n`；完整等间隔采样，时间从零开始。
压缩量表示总厚度减小量，力表示单侧压板测得的力。

```bash
python -m softgriplab demo --output outputs/demo
python -m softgriplab identify fast.csv slow.csv \
  --thickness-m 0.02 --area-m2 0.0004 --noise-std-n 0.005 \
  --output outputs/material.json
```

原有 MPM 平面夹爪和基座加速度脉冲保留在 `mpm-grasp` 中；
Hertz/JKR 参考接触历史可用 `adhesion` 输出为 CSV。

目前的工作重点是模型接入、接触检查、对照算例和可复现数据。完整空间收敛、
试验标定、黏弹性材料向抓取模型的传递仍需继续完善。场景 5 已实现机械臂与弹性软体的分区耦合。
目前没有真实机器人实验或损伤预测验证。[后续安排](docs/ROADMAP.md)。

[GPL-3.0](LICENSE) · [第三方说明](NOTICE)。
