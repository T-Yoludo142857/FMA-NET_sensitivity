# 检测编码器与时空编码器优化报告

## 1. 修改概述

本次实现四项低风险优化，保持原有 FME、MAF、MAR 和 ByteTrack 接口不变：

1. 增强高分辨率多尺度融合。
2. 为时间、空间、联合三条 3D 卷积分支加入自适应权重。
3. 增加显式绝对帧差分分支。
4. 增加小目标辅助热图及其监督损失。

本次没有修改跟踪阈值、MAR 规则、数据集标注文件或已有 checkpoint。

## 2. 高分辨率多尺度融合

修改文件：`lib/models/dladcn_gru.py`

新增 `HighResolutionFusion` 模块。它接收 DLAUp 的四个特征尺度，通道数分别为 16、32、64、128，并执行：

1. 用 `1x1` 卷积将各层通道统一为 16；
2. 将低分辨率特征双线性上采样到最高分辨率；
3. 拼接四个尺度的特征；
4. 使用两个 `3x3` 卷积细化；
5. 将结果残差加回最高分辨率特征。

这样可以在保留小目标精确定位信息的同时，引入深层特征的语义上下文，主要用于改善小目标中心定位和召回率。

## 3. 三条 3D 分支自适应加权

修改文件：`lib/models/dladcn_gru.py`

原有三条分支保持不变：

- 时间分支：`3x1x1`，建模同一位置的跨帧变化；
- 空间分支：`1x3x3`，建模单帧局部结构；
- 联合分支：`3x3x3`，同时建模时间和空间变化。

新增门控网络：三条分支输出先做全局平均池化，再经过两层全连接和 softmax，生成三个片段级权重。分支在原有 `1x1x1` 融合卷积前按权重缩放，三个权重之和为 1。

运动明显的片段可以更多使用时间和联合信息，运动较弱的片段可以保留更多空间信息。

## 4. 显式帧差分分支

修改文件：`lib/models/dladcn_gru.py`

对输入 `B x C x T x H x W` 构造绝对帧差分：

```text
diff[:, :, 0] = 0
diff[:, :, i] = abs(x[:, :, i] - x[:, :, i - 1])，i > 0
```

差分序列经过轻量 3D 卷积编码，再由 sigmoid 门控决定加入多少差分特征。该分支把连续帧之间的微小变化显式提供给网络，同时保留原始 RGB 特征路径，避免完全依赖差分噪声。

## 5. 小目标辅助热图

涉及文件：

- `lib/models/dladcn_gru.py`
- `lib/dataset/coco_icpr.py`
- `lib/dataset/coco_mtb.py`
- `lib/Trainer/ctdet.py`
- `lib/Trainer/ctdet_1.py`

模型新增独立的 `hm_small` 热图头，类别数与普通 `hm` 头相同。

数据集代码新增 `small_hm` 标签。当目标经过输出坐标变换后的面积满足：

```text
目标宽度 x 目标高度 <= small_obj_area
```

就把该目标中心写入小目标热图。默认：

```text
small_obj_area = 256
```

当前 `down_ratio=1` 时，大致对应面积不超过 `16x16` 的目标。

新增损失为：

```text
L = L_original + small_hm_weight x L_focal(hm_small, small_hm)
```

默认 `small_hm_weight=0.5`。

推理时使用两张热图的逐点最大值：

```text
hm = max(sigmoid(hm), sigmoid(hm_small))
```

再进入原有 CenterNet 解码流程。这样辅助头可以提高小目标召回率，同时不替换普通检测头。

## 6. 新增配置

新增参数：

```text
--small_obj_area 256
--small_hm_weight 0.5
```

推荐 SatVideoDT 初始训练命令：

```powershell
python train_satvideoDT.py --small_obj_area 256 --small_hm_weight 0.5
```

SatMTB 目标尺寸差异更大，建议先使用相同参数进行对比，再单独测试 `256`、`400`、`576` 三个面积阈值。

## 7. 旧 checkpoint 兼容性

旧权重不包含以下新增模块参数：

- `high_resolution_fusion`；
- `conv_3d.branch_gate`；
- `conv_3d.diff_branch` 和 `conv_3d.diff_gate`；
- `hm_small`。

仓库的 `load_model` 使用 `strict=False`，因此旧权重中匹配的参数可以正常加载，新模块参数会随机初始化。旧 checkpoint 适合用于初始化微调，但必须重新训练或充分微调后才能观察优化收益。

## 8. 建议实验流程

对比实验时固定 FME、MAR、ByteTrack 阈值、`seqLen`、检测置信度阈值、学习率、训练轮数和数据划分。

检测阶段重点观察：Recall、Precision、False Negatives、False Positives，以及按目标尺寸统计的 Recall/AP。

跟踪阶段重点观察：MOTA、IDF1、ID Switches、False Positives 和 Misses。

建议消融顺序：

1. 原始 baseline；
2. 只加入高分辨率融合；
3. 只加入三分支自适应加权；
4. 只加入帧差分分支；
5. 只加入小目标辅助热图；
6. 四项优化全部加入。

## 9. 验证情况

已对修改后的 Python 文件执行 `py_compile`，语法检查通过；`git diff --check` 未发现空白格式错误。

当前环境中的 PyTorch 前向测试未能在命令执行时间内完成，因此尚未运行完整训练或单 batch 前向验证。建议在配置好的 CUDA/PyTorch 环境中先运行一个 batch 的 forward，再开始完整训练。
