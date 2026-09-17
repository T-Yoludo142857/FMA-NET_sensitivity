"""
================================================================================
FMA-Net 动态参数空间测试框架
Parameter Sensitivity Analysis Framework
================================================================================

用途：系统地测试 FMA-Net 中每个参数对模型性能的影响，确定：
  1. 参数的实际有效范围
  2. 参数对最终跟踪指标的敏感度
  3. 参数间的交互关系（可选）

使用方法：
  python param_sensitivity_framework.py --task [detection|tracking|full] --method [single|grid|random]

作者：自动生成
日期：2026-07-31
================================================================================
"""

import os
import sys
import json
import copy
import argparse
import itertools
import numpy as np
from collections import OrderedDict, defaultdict
from datetime import datetime

# ============================================================================
# 第一部分：完整参数空间定义
# ============================================================================

PARAM_SPACE = {
    # ------------------------------------------------------------------
    # A. 训练参数 (Training)
    # ------------------------------------------------------------------
    "train": OrderedDict({
        "lr": {
            "default": 1.25e-4,
            "range": [1e-5, 5e-4],
            "test_values": [5e-5, 1.25e-4, 2.5e-4, 5e-4],
            "type": "continuous",
            "scale": "log",
            "description": "学习率。控制模型权重更新的步长",
            "expected_impact": "high",
            "expected_effect": "过大会导致训练不稳定/不收敛；过小收敛过慢。影响 MOTA 约 ±3-5%",
            "affected_metrics": ["MOTA", "IDF1", "IDs"],
        },
        "batch_size": {
            "default": 4,
            "range": [1, 16],
            "test_values": [1, 2, 4, 8],
            "type": "discrete",
            "description": "每批次的序列样本数",
            "expected_impact": "medium",
            "expected_effect": "影响 BatchNorm 统计量和梯度稳定性。小 batch 引入更多噪声，大 batch 需要调整 lr",
            "affected_metrics": ["MOTA", "MOTP"],
        },
        "num_epochs": {
            "default": 30,
            "range": [10, 70],
            "test_values": [20, 30, 50, 70],
            "type": "discrete",
            "description": "训练总轮数",
            "expected_impact": "high",
            "expected_effect": "更多轮次通常提升性能，但存在过拟合风险。影响 MOTA 约 ±2-5%",
            "affected_metrics": ["MOTA", "IDF1"],
        },
        "lr_step": {
            "default": [15, 22],
            "range": "dependent",
            "test_values": [[10, 20], [15, 22], [20, 30], [25, 35]],
            "type": "list",
            "description": "学习率下降的 epoch 节点",
            "expected_impact": "medium",
            "expected_effect": "过早下降限制模型学习，过晚可能导致震荡",
            "affected_metrics": ["MOTA", "loss"],
        },
        "scheduler": {
            "default": "cos",
            "range": ["cos", "multi"],
            "test_values": ["cos", "multi"],
            "type": "categorical",
            "description": "学习率调度策略。cos=余弦退火，multi=阶梯下降",
            "expected_impact": "medium",
            "expected_effect": "cos 在卫星视频小目标上通常更平滑稳定",
            "affected_metrics": ["MOTA", "loss"],
        },
        "warmup_iters": {
            "default": 5000,
            "range": [1000, 20000],
            "test_values": [2000, 5000, 10000, 19512],
            "type": "discrete",
            "description": "线性 warmup 迭代次数。逐步将 lr 从 0 升至目标值",
            "expected_impact": "low",
            "expected_effect": "适当 warmup 有助于训练初期稳定性，对最终指标影响较小",
            "affected_metrics": ["loss"],
        },
        "seqLen": {
            "default": 5,
            "range": [2, 8],
            "test_values": [3, 5, 7],
            "type": "discrete",
            "description": "输入序列长度（帧数）。越大捕获越多时序信息，但内存消耗增加",
            "expected_impact": "high",
            "expected_effect": "更长的序列能建模更远的时序依赖，但 GPU 显存线性增长。影响 MOTA 约 ±2-4%",
            "affected_metrics": ["MOTA", "IDF1", "IDs", "GPU_mem"],
        },
        "radius_mapping": {
            "default": 3,
            "range": [1, 7],
            "test_values": [1, 3, 5, 7],
            "type": "discrete",
            "description": "FME 光流位移映射的邻域扩散半径。控制运动信息的空间传播范围",
            "expected_impact": "medium",
            "expected_effect": "太小则运动信息仅覆盖单点，太大则引入噪声。卫星小目标通常取 1-5",
            "affected_metrics": ["MOTA", "IDF1"],
        },
    }),

    # ------------------------------------------------------------------
    # B. 模型结构参数 (Architecture)
    # ------------------------------------------------------------------
    "architecture": OrderedDict({
        "model_name": {
            "default": "DLADCN",
            "range": ["DLADCN"],
            "test_values": ["DLADCN"],
            "type": "categorical",
            "description": "模型架构名称（当前仅支持 DLADCN）",
            "expected_impact": "n/a",
            "expected_effect": "不同 backbone 对检测精度影响显著",
            "affected_metrics": ["MOTA", "FPS"],
        },
        "head_conv": {
            "default": 128,
            "range": [64, 256],
            "test_values": [64, 128, 256],
            "type": "discrete",
            "description": "检测头中卷积层的通道数",
            "expected_impact": "medium",
            "expected_effect": "更多通道增强表达能力但增加计算量",
            "affected_metrics": ["MOTA", "FPS", "GPU_mem"],
        },
        "down_ratio": {
            "default": 1,
            "range": [1, 4],
            "test_values": [1, 2, 4],
            "type": "discrete",
            "description": "输出特征图相对于输入的下采样倍率。1=不降采样",
            "expected_impact": "high",
            "expected_effect": "卫星视频目标极小，降采样会丢失细节。保持 1 最安全",
            "affected_metrics": ["MOTA", "MOTP", "FPS"],
        },
    }),

    # ------------------------------------------------------------------
    # C. 损失函数参数 (Loss)
    # ------------------------------------------------------------------
    "loss": OrderedDict({
        "hm_weight": {
            "default": 1.0,
            "range": [0.5, 5.0],
            "test_values": [0.5, 1.0, 2.0, 5.0],
            "type": "continuous",
            "description": "heatmap 损失权重（Focal Loss）。控制目标中心点定位的重要性",
            "expected_impact": "high",
            "expected_effect": "增大权重使模型更关注目标中心定位精度。影响 MOTA 约 ±2-4%",
            "affected_metrics": ["MOTA", "MOTP"],
        },
        "wh_weight": {
            "default": 0.1,
            "range": [0.01, 1.0],
            "test_values": [0.05, 0.1, 0.5, 1.0],
            "type": "continuous",
            "description": "宽高回归损失权重。控制目标框大小预测的重要性",
            "expected_impact": "medium",
            "expected_effect": "影响边界框精度，进而影响 IoU 匹配质量",
            "affected_metrics": ["MOTP", "MOTA"],
        },
        "off_weight": {
            "default": 1.0,
            "range": [0.5, 5.0],
            "test_values": [0.5, 1.0, 2.0, 5.0],
            "type": "continuous",
            "description": "中心点偏移回归损失权重。修正量化误差",
            "expected_impact": "medium",
            "expected_effect": "影响目标精确定位，对卫星小目标尤为重要",
            "affected_metrics": ["MOTP", "MOTA"],
        },
        "seq_weight": {
            "default": 1.0,
            "range": [0.1, 5.0],
            "test_values": [0.5, 1.0, 2.0, 5.0],
            "type": "continuous",
            "description": "时序 heatmap 序列损失权重。控制多帧 mask 预测的质量",
            "expected_impact": "medium",
            "expected_effect": "影响模型对时序一致性的学习，进而影响光流估计的可靠性",
            "affected_metrics": ["IDF1", "IDs"],
        },
    }),

    # ------------------------------------------------------------------
    # D. 推理/检测参数 (Detection / Inference)
    # ------------------------------------------------------------------
    "detection": OrderedDict({
        "K": {
            "default": 450,
            "range": [100, 1000],
            "test_values": [200, 450, 700, 1000],
            "type": "discrete",
            "description": "每帧检测的最大目标数（top-K）。不足时漏检，过多引入 FP",
            "expected_impact": "high",
            "expected_effect": "卫星视频场景目标密度变化大。太小=漏检↑，太大=FP↑速度↓。影响 MOTA 约 ±3-6%",
            "affected_metrics": ["MOTA", "IDF1", "Recall", "Precision", "FPS"],
        },
        "conf_thres": {
            "default": 0.2,
            "range": [0.05, 0.6],
            "test_values": [0.1, 0.2, 0.3, 0.5],
            "type": "continuous",
            "description": "检测置信度阈值。低于此值的检测被丢弃",
            "expected_impact": "high",
            "expected_effect": "核心 trade-off 参数。低=Recall↑FP↑，高=Precision↑Recall↓。影响 MOTA 约 ±5-10%",
            "affected_metrics": ["MOTA", "IDF1", "Recall", "Precision", "IDs"],
        },
        "nms_thres": {
            "default": 0.4,
            "range": [0.2, 0.7],
            "test_values": [0.3, 0.4, 0.5, 0.6],
            "type": "continuous",
            "description": "NMS IoU 阈值。高于此 IoU 的框被认为重叠",
            "expected_impact": "medium",
            "expected_effect": "低阈值=更多抑制=更少框。密集目标场景需高阈值。影响 MOTA 约 ±1-3%",
            "affected_metrics": ["MOTA", "Precision", "FP"],
        },
        "score_threshold": {
            "default": 0.08,
            "range": [0.01, 0.3],
            "test_values": [0.05, 0.08, 0.15, 0.25],
            "type": "continuous",
            "description": "FME 中 mask 解码时用于筛选目标的分数阈值",
            "expected_impact": "medium",
            "expected_effect": "控制传递给光流追踪的目标质量。过高=跟踪点不足，过低=噪声跟踪点",
            "affected_metrics": ["IDF1", "IDs"],
        },
        "nms_kernel": {
            "default": 3,
            "range": [3, 7],
            "test_values": [3, 5, 7],
            "type": "discrete",
            "description": "heatmap NMS 的核大小（max pooling 窗口）",
            "expected_impact": "low",
            "expected_effect": "影响 heatmap 上的峰值选择密度。越大越稀疏",
            "affected_metrics": ["Precision", "Recall"],
        },
    }),

    # ------------------------------------------------------------------
    # E. 跟踪参数 (Tracking / ByteTrack)
    # ------------------------------------------------------------------
    "tracking": OrderedDict({
        "high_score_thr": {
            "default": 0.3,
            "range": [0.2, 0.7],
            "test_values": [0.25, 0.3, 0.4, 0.5],
            "type": "continuous",
            "description": "ByteTrack 高分匹配阈值。高于此分数进入第一轮匹配",
            "expected_impact": "high",
            "expected_effect": "控制高质量检测参与首次 IoU 匹配的资格。影响 MOTA 约 ±3-5%",
            "affected_metrics": ["MOTA", "IDF1", "IDs"],
        },
        "low_score_thr": {
            "default": 0.2,
            "range": [0.05, 0.4],
            "test_values": [0.1, 0.2, 0.3],
            "type": "continuous",
            "description": "ByteTrack 低分匹配阈值。高/低分之间的检测参与第二轮匹配",
            "expected_impact": "medium",
            "expected_effect": "太低引入大量噪声；太高则丢失遮挡/模糊目标",
            "affected_metrics": ["MOTA", "Recall", "IDs"],
        },
        "init_track_thr": {
            "default": 0.2,
            "range": [0.1, 0.8],
            "test_values": [0.15, 0.2, 0.4, 0.6],
            "type": "continuous",
            "description": "初始化新轨迹的检测分数阈值。高于此值的检测可能开启新轨迹",
            "expected_impact": "high",
            "expected_effect": "低=更多新轨迹=IDs↓FP↑；高=更少新轨迹=FN↑丢失↑。影响 MOTA 约 ±3-5%",
            "affected_metrics": ["IDF1", "IDs", "MOTA", "FN"],
        },
        "match_iou_high": {
            "default": 0.1,
            "range": [0.05, 0.3],
            "test_values": [0.05, 0.1, 0.2, 0.3],
            "type": "continuous",
            "description": "第一轮（高分）匹配的 IoU 阈值",
            "expected_impact": "medium",
            "expected_effect": "卫星视频帧间位移小，低 IoU 阈值即可。过高会导致 ID switch 增加",
            "affected_metrics": ["IDs", "MOTA"],
        },
        "match_iou_low": {
            "default": 0.1,
            "range": [0.05, 0.3],
            "test_values": [0.05, 0.1, 0.2, 0.3],
            "type": "continuous",
            "description": "第二轮（低分）匹配的 IoU 阈值",
            "expected_impact": "low",
            "expected_effect": "辅助参数，影响低置信度检测的匹配",
            "affected_metrics": ["IDs", "Recall"],
        },
        "match_iou_tentative": {
            "default": 0.1,
            "range": [0.05, 0.3],
            "test_values": [0.05, 0.1, 0.2, 0.3],
            "type": "continuous",
            "description": "未确认/tentative 轨迹匹配的 IoU 阈值",
            "expected_impact": "low",
            "expected_effect": "影响新生轨迹的确认速度",
            "affected_metrics": ["IDs", "FN"],
        },
        "num_tentatives": {
            "default": 2,
            "range": [1, 5],
            "test_values": [1, 2, 3, 5],
            "type": "discrete",
            "description": "确认新轨迹需要的连续检测帧数",
            "expected_impact": "medium",
            "expected_effect": "值越低确认越快但 FP 轨迹风险↑。卫星视频目标慢速运动，2-3 较合适",
            "affected_metrics": ["IDF1", "IDs", "FP"],
        },
        "num_frames_retain": {
            "default": 30,
            "range": [10, 120],
            "test_values": [15, 30, 60, 120],
            "type": "discrete",
            "description": "轨迹消失后保留的最大帧数（用于可能的重新连接）",
            "expected_impact": "medium",
            "expected_effect": "较长的保留有助于处理短期遮挡，但增加 ID switch 风险",
            "affected_metrics": ["IDF1", "IDs"],
        },
        "vdc_weight": {
            "default": 0.2,
            "range": [0.0, 0.5],
            "test_values": [0.0, 0.1, 0.2, 0.5],
            "type": "continuous",
            "description": "速度方向一致性（VDC）在匹配成本中的权重",
            "expected_impact": "medium",
            "expected_effect": "利用运动方向改善匹配，对线性运动目标有效。0=仅用 IoU",
            "affected_metrics": ["IDF1", "IDs"],
        },
    }),

    # ------------------------------------------------------------------
    # F. 运动感知优化参数 (MAR / Post-processing)
    # ------------------------------------------------------------------
    "mar": OrderedDict({
        "cost_limit": {
            "default": 0.9,
            "range": [0.5, 1.5],
            "test_values": [0.7, 0.9, 1.1, 1.3],
            "type": "continuous",
            "description": "MAR 中 lapjv 匹配的成本上限。超过此值认为不匹配",
            "expected_impact": "medium",
            "expected_effect": "控制 MAR 中检测框修正的匹配严格度",
            "affected_metrics": ["MOTA", "IDF1"],
        },
    }),
}

# ============================================================================
# 第二部分：测试用例生成
# ============================================================================


def flatten_params(param_space, categories=None):
    """将参数空间扁平化为 (param_name, config) 列表"""
    flat = []
    for cat, params in param_space.items():
        if categories and cat not in categories:
            continue
        for pname, pconfig in params.items():
            flat.append((f"{cat}.{pname}", pconfig))
    return flat


def generate_single_param_tests(param_space, categories=None):
    """
    单参数变化测试：每次只变化一个参数，其他保持默认。
    返回 [{param_full_name, param_config, test_value, default_config}, ...]
    """
    tests = []
    for cat, params in param_space.items():
        if categories and cat not in categories:
            continue
        for pname, pconfig in params.items():
            for val in pconfig["test_values"]:
                tests.append({
                    "vary_param": f"{cat}.{pname}",
                    "vary_config": pconfig,
                    "test_value": val,
                    "is_default": val == pconfig["default"],
                    "category": cat,
                })
    return tests


def generate_grid_tests(param_space, params_to_test, categories=None):
    """网格搜索：对指定参数做全组合。注意组合爆炸。"""
    selected = []
    for full_name, pconfig in flatten_params(param_space, categories):
        if full_name in params_to_test or params_to_test == ["all"]:
            selected.append((full_name, pconfig["test_values"]))
    if not selected:
        raise ValueError("No matching parameters found for grid search")

    names, value_lists = zip(*selected)
    for combo in itertools.product(*value_lists):
        yield dict(zip(names, combo))


def generate_random_tests(param_space, n_samples=50, categories=None):
    """随机采样参数空间（用于探索性测试）"""
    import random
    random.seed(42)

    flat = flatten_params(param_space, categories)
    tests = []
    for _ in range(n_samples):
        config = {}
        for pname, pconfig in flat:
            if pconfig["type"] == "categorical":
                config[pname] = random.choice(pconfig["test_values"])
            elif pconfig["type"] == "continuous":
                lo, hi = pconfig["range"]
                if pconfig.get("scale") == "log":
                    config[pname] = 10 ** random.uniform(np.log10(lo), np.log10(hi))
                else:
                    config[pname] = random.uniform(lo, hi)
            elif pconfig["type"] == "discrete":
                config[pname] = random.choice(pconfig["test_values"])
            elif pconfig["type"] == "list":
                config[pname] = random.choice(pconfig["test_values"])
        tests.append(config)
    return tests


# ============================================================================
# 第三部分：实验执行器
# ============================================================================


class SensitivityExperimentRunner:
    """参数敏感性实验的执行引擎"""

    def __init__(self, opt_class, base_config=None):
        """
        Args:
            opt_class: FMA-Net 的 opts 类
            base_config: 基础配置字典，覆盖默认值
        """
        self.opt_class = opt_class
        self.base_config = base_config or {}
        self.results = []

    def build_config(self, param_overrides):
        """
        根据参数覆盖构建完整的 opt 配置。
        param_overrides: {"train.lr": 2.5e-4, "detection.K": 700, ...}
        """
        opt = self.opt_class().parse(args=[])
        # 应用基础配置
        for k, v in self.base_config.items():
            setattr(opt, k, v)
        # 应用参数覆盖
        for full_name, value in param_overrides.items():
            category, pname = full_name.split(".", 1)
            # 映射到 opt 的属性名
            attr_name = self._map_to_attr(category, pname)
            if hasattr(opt, attr_name):
                setattr(opt, attr_name, value)
            else:
                print(f"  [WARN] opt has no attribute '{attr_name}', storing as custom")
        return opt

    def _map_to_attr(self, category, pname):
        """将参数分类名映射到 opts 的属性名"""
        # 直接映射表
        mapping = {
            "train.lr": "lr",
            "train.batch_size": "batch_size",
            "train.num_epochs": "num_epochs",
            "train.lr_step": "lr_step",
            "train.seqLen": "seqLen",
            "train.radius_mapping": "radius_mapping",
            "train.scheduler": "scheduler",
            "train.warmup_iters": "warmup_iters",
            "architecture.down_ratio": "down_ratio",
            "architecture.model_name": "model_name",
            "detection.K": "K",
            "detection.conf_thres": "conf_thres",
            "detection.nms_thres": "nms_thres",
        }
        return mapping.get(f"{category}.{pname}", pname)

    def run_single_experiment(self, param_overrides, dry_run=True):
        """
        运行单次实验（或干运行，仅打印配置）。

        在完整模式下，这将：
        1. 构建配置
        2. 训练模型（或加载预训练权重）
        3. 在验证集上推理
        4. 运行评估
        5. 收集指标

        当前为框架版本，输出实验计划。实际运行时取消 dry_run=True。
        """
        opt = self.build_config(param_overrides)

        if dry_run:
            return {"status": "dry_run", "config": param_overrides}

        # ---- 实际运行代码 ----
        # 此处接入实际的训练/推理/评估流程
        # 例如：
        # from train_satmtb import main
        # metrics = main(opt)
        # return metrics
        raise NotImplementedError(
            "Set dry_run=False and implement the actual training/evaluation pipeline"
        )

    def run_sensitivity_sweep(self, param_space, categories=None, dry_run=True):
        """运行完整的单参数敏感性扫描"""
        tests = generate_single_param_tests(param_space, categories)
        print(f"\n{'='*70}")
        print(f"单参数敏感性扫描: 共 {len(tests)} 个测试点")
        print(f"{'='*70}\n")

        results_by_param = defaultdict(list)
        default_result = None

        for i, test in enumerate(tests):
            full_name = test["vary_param"]
            value = test["test_value"]
            is_default = test["is_default"]

            # 构建参数覆盖（只覆盖变化的参数）
            overrides = {full_name: value}

            print(f"[{i+1}/{len(tests)}] {full_name} = {value}"
                  f"{' (DEFAULT)' if is_default else ''}")

            result = self.run_single_experiment(overrides, dry_run=dry_run)
            result["param"] = full_name
            result["value"] = value
            result["is_default"] = is_default

            if is_default:
                default_result = result

            results_by_param[full_name].append(result)

        return results_by_param, default_result

    def generate_report(self, results_by_param, output_path=None):
        """生成 Markdown 格式的敏感性分析报告"""
        lines = []
        lines.append("# FMA-Net 参数敏感性分析报告")
        lines.append(f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        lines.append("## 参数空间总览\n")
        lines.append(f"| 类别 | 参数数 | 测试点数 |")
        lines.append(f"|------|--------|----------|")
        for cat, params in PARAM_SPACE.items():
            n_params = len(params)
            n_tests = sum(len(p["test_values"]) for p in params.values())
            lines.append(f"| {cat} | {n_params} | {n_tests} |")

        lines.append(f"\n**总参数数：{sum(len(p) for p in PARAM_SPACE.values())}**")
        lines.append(f"**总测试点数：{sum(len(r) for r in results_by_param.values())}**")

        lines.append("\n---\n")
        lines.append("## 高影响力参数（expected_impact = high）\n")

        for cat, params in PARAM_SPACE.items():
            for pname, pconfig in params.items():
                if pconfig["expected_impact"] == "high":
                    lines.append(f"### {cat}.{pname}")
                    lines.append(f"- **默认值**: `{pconfig['default']}`")
                    lines.append(f"- **测试范围**: `{pconfig['test_values']}`")
                    lines.append(f"- **类型**: {pconfig['type']}")
                    lines.append(f"- **预期影响**: {pconfig['expected_effect']}")
                    lines.append(f"- **影响指标**: {', '.join(pconfig['affected_metrics'])}")
                    lines.append("")

        lines.append("---\n")
        lines.append("## 低影响力参数（expected_impact = low）\n")
        lines.append("以下参数对实验结果影响较小，可作为调参的次要对象：\n")
        for cat, params in PARAM_SPACE.items():
            for pname, pconfig in params.items():
                if pconfig["expected_impact"] == "low":
                    lines.append(f"- **{cat}.{pname}**: {pconfig['description']}（默认 `{pconfig['default']}`）")

        lines.append("\n---\n")
        lines.append("## 推荐测试优先级\n")
        lines.append("按预期影响力排序的测试优先级：\n")
        lines.append("### 第一优先级（高影响 + 大范围）")
        priority1 = []
        for cat, params in PARAM_SPACE.items():
            for pname, pconfig in params.items():
                if pconfig["expected_impact"] == "high":
                    priority1.append(f"- **{cat}.{pname}**: 默认=`{pconfig['default']}`, "
                                     f"测试={pconfig['test_values']}")
        lines.extend(priority1)

        lines.append("\n### 第二优先级（中影响）")
        priority2 = []
        for cat, params in PARAM_SPACE.items():
            for pname, pconfig in params.items():
                if pconfig["expected_impact"] == "medium":
                    priority2.append(f"- **{cat}.{pname}**: 默认=`{pconfig['default']}`, "
                                     f"测试={pconfig['test_values']}")
        lines.extend(priority2)

        report = "\n".join(lines)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n报告已保存至: {output_path}")

        return report


# ============================================================================
# 第四部分：CLI & Demo
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="FMA-Net 参数敏感性分析框架"
    )
    parser.add_argument("--task", type=str, default="report",
                        choices=["report", "sweep", "grid", "random"],
                        help="report=仅生成参数空间报告；sweep=单参数扫描（干运行）；"
                             "grid=网格搜索；random=随机采样")
    parser.add_argument("--categories", type=str, nargs="+",
                        default=None,
                        choices=list(PARAM_SPACE.keys()),
                        help="限定测试的参数类别，默认全部")
    parser.add_argument("--method", type=str, default="single",
                        choices=["single", "grid", "random"])
    parser.add_argument("--n_samples", type=int, default=30,
                        help="随机采样的样本数")
    parser.add_argument("--output", type=str, default="param_sensitivity_report.md",
                        help="报告输出路径")
    parser.add_argument("--dry_run", action="store_true", default=True,
                        help="干运行模式（仅生成计划，不实际执行训练）")

    args = parser.parse_args()

    runner = SensitivityExperimentRunner(None)  # 不需要 opt_class 用于 report

    if args.task == "report":
        # 仅生成参数空间文档
        results_by_param = {}
        for cat, params in PARAM_SPACE.items():
            if args.categories and cat not in args.categories:
                continue
            for pname, pconfig in params.items():
                results_by_param[f"{cat}.{pname}"] = [
                    {"param": f"{cat}.{pname}", "value": v,
                     "is_default": v == pconfig["default"]}
                    for v in pconfig["test_values"]
                ]
        report = runner.generate_report(results_by_param, args.output)
        print(report)

    elif args.task == "sweep":
        # 干运行：打印所有测试计划
        tests = generate_single_param_tests(PARAM_SPACE, args.categories)
        print(f"\n共 {len(tests)} 个测试点：\n")
        header = f"{'#':<5} {'Parameter':<35} {'TestValue':<15} {'Default':<8} {'Category'}"
        print(header)
        print("-" * len(header))
        for i, t in enumerate(tests):
            is_def = "YES" if t["is_default"] else ""
            print(f"{i+1:<5} {t['vary_param']:<35} {str(t['test_value']):<15} "
                  f"{is_def:<8} {t['category']}")

        # 统计
        high_impact = [t for t in tests
                       if t["vary_config"]["expected_impact"] == "high"]
        print(f"\nHigh-impact param tests: {len(high_impact)}")
        print(f"Recommended priority test params:")
        seen = set()
        for t in high_impact:
            if t["vary_param"] not in seen:
                print(f"  - {t['vary_param']} (默认={t['vary_config']['default']}, "
                      f"影响={t['vary_config']['expected_effect'].split('.')[0]})")
                seen.add(t["vary_param"])

    elif args.task == "random":
        tests = generate_random_tests(PARAM_SPACE, args.n_samples, args.categories)
        print(f"\n随机采样 {len(tests)} 组参数配置：\n")
        for i, config in enumerate(tests[:10]):  # 只显示前10组
            print(f"--- 配置 {i+1} ---")
            for k, v in config.items():
                print(f"  {k}: {v}")
        if len(tests) > 10:
            print(f"  ... (共 {len(tests)} 组)")

    print(f"\n{'='*70}")
    print("下一步：准备实际测试环境")
    print(f"{'='*70}")
    print("""
实际运行测试的步骤：
1. 安装依赖：pip install -r requirements.txt
2. 准备数据：将 SatMTB/SatVideoDT 放入 ./data/ 目录
3. 下载预训练模型：从 Google Drive 放入 ./checkpoints/
4. 将框架脚本中的 dry_run=True 改为 dry_run=False
5. 接入实际的训练/推理/评估流水线
6. 运行：python param_sensitivity_framework.py --task sweep
""")


if __name__ == "__main__":
    main()
