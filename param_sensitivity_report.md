# FMA-Net 参数敏感性分析报告

> **实验日期**：2026-08-02
> **测试环境**：gpu01 (4×RTX 4090 24GB, 125GB RAM), CentOS 7, Python 3.8, PyTorch 1.12+cu116
> **测试数据集**：SatVideoDT (ICPR), val split, 11 sequences, 281,165 total objects
> **预训练模型**：SatvideoDT.pth (epoch 20)

---

## 1. 测试方法

对每个参数，**保持其他参数不变，仅改变目标参数值**，依次执行：

1. `test_satvideoDT.py` → 生成跟踪输出 .txt 文件
2. `eval_satvideoDT.py` → 对比 GT 计算 MOTA / IDF1 / IDs / MOTP 等指标
3. 与基准线对比，分析参数影响

**测试范围**：仅推理阶段参数（不改模型权重），训练参数需重新训练暂不覆盖。

---

## 2. 参数空间总览

| 类别 | 参数数 | 本次测试 | 说明 |
|------|--------|----------|------|
| train | 8 | 1 (seqLen) | 训练参数需重新训练，暂不覆盖 |
| architecture | 3 | 0 | 需重新训练 |
| loss | 4 | 0 | 需重新训练 |
| detection | 5 | 3 (conf_thres, K, nms_thres) | 命令行参数可直接控制 |
| tracking | 9 | 4 (high/low/init_track_thr, match_iou) | 需在 test 脚本中修改 |
| mar | 1 | 1 (cost_limit) | 在 mar.py 中修改 |

**实际可测试参数：8 个，约 24 个测试点**

---

## 3. 基准线 (Baseline)

```
conf_thres=0.2, K=450, seqLen=3, radius_mapping=3
ByteTrack: high=0.3, low=0.2, init_track=0.2, match_iou=0.1, tentatives=2
```

| 指标 | 值 | 说明 |
|------|-----|------|
| **MOTA** | 19.9% | 综合跟踪准确率 (Multiple Object Tracking Accuracy) |
| **MOTP** | 0.497 | 定位精度 (Multiple Object Tracking Precision) |
| **IDF1** | 34.8% | 身份保持 F1 得分 |
| **Recall** | 23.0% | 目标召回率 |
| **Precision** | 89.4% | 检测精确率 |
| **IDs** | 903 | ID 切换次数 |
| **FP** | 7,644 | 误报数 |
| **FN** | 216,633 | 漏检数 |
| **MT** | 244 | 大部分被跟踪 (Mostly Tracked) |
| **ML** | 831 | 大部分丢失 (Mostly Lost) |

---

## 4. 参数测试结果

### 4.1 detection.conf_thres (置信度阈值)

| conf_thres | MOTA | IDF1 | MOTP | Recall | Precision | IDs | FP | FN |
|------------|------|------|------|--------|-----------|-----|-----|-----|
| 0.1 | 19.6% | 34.0% | 0.496 | 22.4% | 90.0% | 955 | 6,998 | 218,207 |
| **0.2 (基准)** | **19.9%** | **34.8%** | **0.497** | **23.0%** | **89.4%** | **903** | **7,644** | **216,633** |
| 0.3 | 18.1% | 31.5% | NaN | 20.3% | 91.7% | 935 | 5,171 | 224,109 |

**分析**：conf_thres=0.2 三者中 MOTA 最佳。阈值降低至 0.1 引入噪声 FP 增加但 Recall 未改善；阈值提升至 0.3 丢失大量有效检测导致 FN↑ Recall↓。0.2 为最佳平衡点。

### 4.2 detection.K (最大检测数)

| K | MOTA | IDF1 | MOTP | Recall | Precision | IDs | FP | FN |
|---|------|------|------|--------|-----------|-----|-----|-----|
| 200 | 19.9% | 34.8% | 0.497 | 23.0% | 89.4% | 903 | 7,644 | 216,633 |
| **450 (基准)** | **19.9%** | **34.8%** | **0.497** | **23.0%** | **89.4%** | **903** | **7,644** | **216,633** |
| 700 | 19.9% | 34.8% | 0.497 | 23.0% | 89.4% | 903 | 7,644 | 216,633 |

**分析**：K=200/450/700 三组结果完全相同。SatVideoDT 卫星视频目标稀疏，每帧实际检测数远小于 200，因此增大 K 无任何影响。此参数在此数据集上可忽略。

### 4.3 tracking.high_score_thr (高分匹配阈值)

| high_score_thr | MOTA | IDF1 | MOTP | Recall | Precision | IDs | FP | FN |
|----------------|------|------|------|--------|-----------|-----|-----|-----|
| **0.25** | **25.8%** | **42.9%** | **0.508** | **31.7%** | **85.1%** | **1,161** | **15,675** | **191,915** |
| 0.35 (基准) | 19.9% | 34.8% | 0.497 | 23.0% | 89.4% | 903 | 7,644 | 216,633 |
| 0.45 | 5.1% | 11.7% | 0.485 | 6.7% | 83.8% | 658 | 3,616 | 262,463 |

**分析**：此参数极敏感，MOTA 从 5.1%→25.8% 跨度达 20.7%。卫星目标检测置信度整体偏低，降低阈值至 0.25 让更多有效检测进入第一轮高质量匹配。0.35 以上阈值过高导致高质量匹配池几乎为空，性能骤降。**推荐值：0.25**。

### 4.4 tracking.init_track_thr (轨迹初始化阈值)

> 测试基于 high_score_thr=0.25（当前最优）

| init_track_thr | MOTA | IDF1 | MOTP | Recall | Precision | IDs | FP | FN |
|----------------|------|------|------|--------|-----------|-----|-----|-----|
| 0.15 | 25.8% | 42.9% | 0.508 | 31.7% | 85.1% | 1,161 | 15,675 | 191,915 |
| **0.2 (基准)** | **25.8%** | **42.9%** | **0.508** | **31.7%** | **85.1%** | **1,161** | **15,675** | **191,915** |
| 0.4 | 25.8% | 42.9% | NaN | 31.7% | 85.2% | 1,154 | 15,482 | 192,062 |
| 0.6 | 3.7% | 7.8% | NaN | 4.4% | 87.1% | 201 | 1,819 | 268,851 |

**分析**：0.15-0.4 区间完全不敏感，结果几乎相同；0.6 阈值过高导致绝大多数检测无法初始化新轨迹，FN 爆炸。**推荐保持 0.2**。

### 4.5 train.radius_mapping (光流邻域扩散半径)

> 测试基于 high_score_thr=0.25, init_track_thr=0.2

| radius_mapping | MOTA | IDF1 | MOTP | Recall | Precision | IDs | FP | FN |
|----------------|------|------|------|--------|-----------|-----|-----|-----|
| 1 | 25.8% | 43.0% | 0.508 | 31.8% | 85.1% | 1,176 | 15,612 | 191,824 |
| **3 (基准)** | **25.8%** | **42.9%** | **0.508** | **31.7%** | **85.1%** | **1,161** | **15,675** | **191,915** |
| 5 | 25.7% | 42.9% | 0.509 | 31.7% | 85.1% | 1,178 | 15,668 | 191,987 |

**分析**：1/3/5 三组 MOTA 最大差异仅 0.1%，此参数对模型结果几乎无影响。光流位移的邻域扩散半径在此数据集上不敏感，可能因为卫星目标太小、帧间位移极小。**推荐保持默认值 3**。

### 4.6 测试结果汇总

| 参数 | 最优值 | 默认值 | MOTA 提升 | 敏感度 |
|------|--------|--------|-----------|--------|
| **high_score_thr** | **0.25** | 0.35 | **+5.9% (19.9%→25.8%)** | 🔴 极高 |
| conf_thres | 0.2 | 0.2 | 0%（默认即最优） | 🔴 高 |
| init_track_thr | 0.15-0.4 | 0.2 | 0%（区间内不敏感） | 🟡 中（仅在极端值时触发） |
| K | 任意 | 450 | 0%（无效参数） | 🟢 无 |
| radius_mapping | 1-5 | 3 | 0%（无效参数） | 🟢 无 |

---

## 5. 结论与分析

### 5.1 高敏感度参数

**`high_score_thr` 是唯一有显著影响的参数**，从默认 0.35 降至 0.25 带来 **MOTA +5.9%，IDF1 +8.1%**。原因：卫星视频目标检测置信度偏低（多在 0.2-0.4 区间），默认 0.35 阈值将大量正确检测排除在高分匹配池外，导致 ByteTrack 第一轮高质量匹配退化。

### 5.2 低/无敏感度参数

- **K**（最大检测数）：卫星目标稀疏，每帧检测数 < 200，K 在 200-700 范围内无影响
- **radius_mapping**（光流邻域半径）：FME 的光流扩散范围在 1-5 像素内无差异，卫星目标帧间位移极小
- **init_track_thr**：在 0.15-0.4 区间内完全不敏感，仅极端值 0.6 会破坏性能
- **conf_thres**：0.2 为最优，偏离 ±0.1 性能下降但幅度有限（~2% MOTA）

### 5.3 推荐配置

```
# 最优推理参数
conf_thres     = 0.2    (默认)
K              = 450    (默认，可任意)
high_score_thr = 0.25   (▼ 从 0.35 调低，关键改动)
init_track_thr = 0.2    (默认)
radius_mapping = 3      (默认)
```

| 指标 | 默认配置 | 最优配置 | 提升 |
|------|----------|----------|------|
| MOTA | 19.9% | **25.8%** | **+5.9%** |
| IDF1 | 34.8% | **42.9%** | **+8.1%** |
| Recall | 23.0% | **31.7%** | **+8.7%** |
| MT | 244 | **319** | **+75** |
| ML | 831 | **661** | **−170** |

### 5.4 局限性

- 仅测试了 SatVideoDT 单类数据集，结论在其他数据集上可能不同
- 训练参数（lr, num_epochs, loss weights 等）未测试，需重新训练
- seqLen 已完成测试：seqLen=5 完整复现论文 69.0%（见第 6 节），此前 seqLen=3 是差距主因
- SatMTB 多类数据集未测试

---

## 6. seqLen=5 复现论文结果

> 复现日期：2026-09-17。修复内存问题（换空闲 GPU）后，用论文默认配置重新推理，完整复现论文 Table I。

**配置**：seqLen=5, radius_mapping=3, high=0.35, low=0.2, init_track=0.2, conf_thres=0.2, K=450，权重 SatvideoDT.pth。

| 指标 | 论文 (Table I) | 复现结果 | 一致 |
|------|----------------|----------|------|
| **MOTA** | 69.0% | **69.0%** | ✅ |
| **IDF1** | 76.2% | **76.2%** | ✅ |
| **MT** | 886 | **886** | ✅ |
| **ML** | 124 | **124** | ✅ |
| Rcll | — | 75.0% | — |
| Prcn | — | 93.0% | — |
| IDs | — | 1,059 | — |
| FP | — | 15,797 | — |
| FN | — | 70,368 | — |
| MOTP | — | 0.402 | — |

**结论**：MOTA / IDF1 / MT / ML 四项核心指标与论文完全一致，证明此前 seqLen=3 的 19.9% 与论文 69.0% 的差距**完全由 seqLen=3 导致**——它破坏了 MAR 模块依赖的 5 帧时序窗口，而非模型或参数问题。seqLen=5 + high=0.35（论文默认）即可 100% 复现。

---

---

*报告完成于 2026-08-03*

---

## 附录：评估结果截图

### 测试 1：conf_thres = 0.1

![b3e4dce86947d00b16a5c6414219255d](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\b3e4dce86947d00b16a5c6414219255d.png)

### 测试 2：conf_thres = 0.3

![21895722e6736deb6672ededf7e4b555](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\21895722e6736deb6672ededf7e4b555.png)

### 测试 3：K = 200

![67ddf9d16f9da95f5272132530ad89c1](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\67ddf9d16f9da95f5272132530ad89c1.png)

### 测试 4：K = 450

![d61167f4968f6417940563c62fec8054](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\d61167f4968f6417940563c62fec8054.png)

### 测试 5：K = 700

![f9961d6467d1c4ff9287de85c25597b4](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\f9961d6467d1c4ff9287de85c25597b4.png)

### 测试 6：high_score_thr = 0.25

![fe0df9b7e61ea8363389523a69b7a9cb](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\fe0df9b7e61ea8363389523a69b7a9cb.png)

### 测试 7：high_score_thr = 0.45

![bd9f687f7685b9938c65682b62b670d5](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\bd9f687f7685b9938c65682b62b670d5.png)

### 测试 8：init_track_thr = 0.15

![c738465a0976374334de25f893861033](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\c738465a0976374334de25f893861033.png)

### 测试 9：init_track_thr = 0.4

![ae179e4cb8624b9476f113f9c268d4e8](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\ae179e4cb8624b9476f113f9c268d4e8.png)

### 测试 10：init_track_thr = 0.6

![dbe23624b3d0cfa2f2fc40da5ff7100a](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\dbe23624b3d0cfa2f2fc40da5ff7100a.png)

### 测试 11：radius_mapping = 1

![008d2806a53359524814457d7739ebd8](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\008d2806a53359524814457d7739ebd8.png)

### 测试 12：radius_mapping = 5

![b14001f225b1fa0255b1c046cbc0f1db](C:\Users\24122\OneDrive\xwechat_files\wxid_bfhh2s1rcoio22_fa2b\temp\RWTemp\2026-08\b14001f225b1fa0255b1c046cbc0f1db.png)

