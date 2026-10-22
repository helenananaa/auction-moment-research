from __future__ import annotations

from .predictor import PredictionResult


ISSUE_LABELS = {
    "pre_bid_not_confirmed": "尚未确认处于本轮出价前",
    "exact_map_rows_not_confirmed": "地图总行数尚未确认",
    "visible_map_not_reviewed": "本轮可见地图尚未扫描或复核",
    "pre_bid_visual_state_machine": "出价前时点由视觉状态机判断",
    "map_rows_auto_estimated": "地图行数为自动估算",
    "visible_map_auto_scanned_not_human_reviewed": "地图由自动扫描生成，尚未人工复核",
    "no_compatible_public_world": "当前事件与地图条件组合没有兼容世界",
    "v2_conflict_using_v6_fallback": "v2 条件冲突，已改用 v6 保守回退",
    "v2_unreliable_posterior_using_v6_fallback": "v2 推断支持不足，已回退到 v6",
    "posterior_not_converged": "条件采样尚未收敛",
    "numeric_constraints_infeasible": "当前数值条件在生成模型支持范围内无解",
    "unsupported_evidence": "存在当前推断暂不支持的证据",
    "ocr_adapter_not_packet_equivalent": "OCR 证据不等价于结构化协议输入",
    "packet_capture_inputs": "输入来自被动抓包（仅出价前可见的事件与揭示）",
    "interval_uncalibrated": "当前区间尚未校准",
    "missing_event": "存在尚未识别的轮次事件",
    "map_completeness_not_confirmed": "地图完整性尚未确认",
    "compatible_worlds_below_10": "兼容世界少于 10 个",
    "conditioned_resample_provisional": "v2 已按当前条件重新生成候选世界，结果仍属未校准估计",
    "exact_posterior_low_ess": "精确条件后验有效样本不足或计算被时间预算截断",
    "exact_unavailable_using_v6_fallback": "精确条件采样不可用，已回退到 v6",
    "exact_unsupported_evidence": "存在精确条件采样暂不支持或相互冲突的证据",
    "exact_sampling_failed": "精确条件采样运行失败",
    "exact_no_posterior_support": "精确条件采样在当前条件下没有支持",
}

EFFECT_LABELS = {
    "total_item_count": "本局藏品总数",
    "quality_item_count": "指定品质数量",
    "size_item_count": "指定尺寸数量",
    "quality_total_value": "指定品质总价值",
    "size_total_value": "指定尺寸总价值",
    "quality_average_value": "指定品质平均价值",
    "size_average_value": "指定尺寸平均价值",
    "quality_total_area": "指定品质总面积",
    "size_total_area": "指定尺寸总面积",
    "total_occupied_cells": "总占用格数",
    "quality_average_area": "指定品质平均面积",
    "size_average_area": "指定尺寸平均面积",
    "average_item_area": "平均藏品面积",
    "max_item_value": "单件最高价值",
    "max_value_per_cell": "最高每格价值",
    "highest_quality": "最高品质",
}


def describe_issue(value: str) -> str:
    issue = str(value)
    known = ISSUE_LABELS.get(issue)
    if known:
        return known
    if issue.startswith("low_confidence_event:"):
        _, round_number, kind = issue.split(":", 2)
        kind_label = "公共" if kind == "public" else "个人"
        return f"{round_number} {kind_label}事件 OCR 置信度不足"
    if issue.startswith("unparsed_event:"):
        _, round_number, kind = issue.split(":", 2)
        kind_label = "公共" if kind == "public" else "个人"
        return f"{round_number} {kind_label}事件无法解析"
    return issue


def describe_diagnostic(value: str) -> str:
    diagnostic = str(value)
    if diagnostic.startswith("particle_collapse:"):
        detail = describe_diagnostic(diagnostic.removeprefix("particle_collapse:"))
        return f"初始有限世界样本耗尽（不代表条件不可能）：{detail}"
    parts = diagnostic.split(":")
    if parts[:1] == ["conditioned_resample"] and len(parts) == 3:
        return (
            f"已额外生成 {int(parts[1]):,} 个候选世界，"
            f"找到 {int(parts[2]):,} 个兼容世界"
        )
    if parts[:1] == ["conditioned_resample_exhausted"] and len(parts) == 2:
        return f"额外生成 {int(parts[1]):,} 个候选世界后仍无兼容结果"
    if len(parts) in {4, 5} and parts[0] == "event_conflict":
        _, round_number, kind, effect = parts[:4]
        kind_label = "公共" if kind == "public" else "个人"
        observed = f"={parts[4]}" if len(parts) == 5 else ""
        return (
            f"加入 {round_number} {kind_label}事件"
            f"（{EFFECT_LABELS.get(effect, effect)}{observed}）后兼容世界归零"
        )
    if parts[:2] == ["map_conflict", "visible_count"] and len(parts) == 3:
        return f"要求至少可见 {parts[2]} 件藏品后兼容世界归零"
    if parts[:2] == ["map_conflict", "identity"] and len(parts) == 4:
        return f"要求身份 {parts[2]} 至少 {parts[3]} 件后兼容世界归零"
    if parts[:2] == ["map_conflict", "quality"] and len(parts) == 4:
        return f"要求{parts[2]}品质至少 {parts[3]} 件后兼容世界归零"
    if parts[:2] == ["map_conflict", "size"] and len(parts) == 4:
        return f"要求 {parts[2]} 尺寸至少 {parts[3]} 件后兼容世界归零"
    if parts[:2] == ["map_conflict", "quality_size"] and len(parts) == 5:
        return f"要求{parts[2]}品质且 {parts[3]} 尺寸至少 {parts[4]} 件后兼容世界归零"
    if parts[:2] == ["map_conflict", "unknown_identity"] and len(parts) == 3:
        return f"图鉴中不存在自动识别身份 {parts[2]}"
    if parts[:2] == ["map_conflict", "unknown_quality"] and len(parts) == 3:
        return f"无法识别品质 {parts[2]}"
    if parts[:2] == ["map_conflict", "unknown_size"] and len(parts) == 3:
        return f"图鉴不支持尺寸 {parts[2]}"
    return diagnostic


def format_prediction(result: PredictionResult) -> str:
    issues = "；".join(describe_issue(value) for value in result.issues)
    diagnostics = "；".join(describe_diagnostic(value) for value in result.diagnostics)
    if result.p50 is None:
        lines = [
            f"状态：{result.status}",
            "当前无法给出估值或建议出价",
            f"需要处理：{issues or '证据不足'}",
        ]
        if diagnostics:
            lines.append(f"冲突定位：{diagnostics}")
        return "\n".join(lines)

    if result.estimate_source == "v6_default":
        lines = [
            "状态：使用 v6 估值；v2 仅供诊断",
            f"v6 P10 / P50 / P90：{result.p10:,} / {result.p50:,} / {result.p90:,}",
        ]
        if result.v2_prediction is not None:
            lines.append(f"v2 诊断估值（未参与融合）：{result.v2_prediction:,}")
    elif result.estimate_source == "exact_v2_posterior":
        lines = [
            "状态：精确条件后验估值（实验模式）"
            + ("" if result.status == "exact_v2_posterior_estimate" else "；有效样本不足，谨慎参考"),
        ]
        if result.point_estimate is not None:
            lines.append(f"估值（期望相对误差最小点）：{result.point_estimate:,}")
        lines.append(f"后验 P10 / P50 / P90：{result.p10:,} / {result.p50:,} / {result.p90:,}")
        if result.v6_prediction is not None:
            lines.append(f"v6 对照（未参与）：{result.v6_prediction:,}")
    elif result.estimate_source == "v6_fallback":
        heading = (
            "状态：v2 条件冲突，显示 v6 保守回退"
            if result.status == "v6_fallback_v2_conditioning_conflict"
            else "状态：精确条件采样不可用，显示 v6 回退估值"
            if result.status == "v6_fallback_exact_unavailable"
            else "状态：v2 推断支持不足，显示 v6 回退估值"
        )
        lines = [
            heading,
            f"v6 P10 / P50 / P90：{result.p10:,} / {result.p50:,} / {result.p90:,}",
        ]
    else:
        lines = ["状态：已完成本轮估值与手动出价建议"]
        if result.v6_prediction is not None and result.v2_prediction is not None:
            lines.append(
                f"v6 / v2：{result.v6_prediction:,} / {result.v2_prediction:,}"
                f"（v2 权重 {result.generation_weight:.0%}）"
            )
        lines.append(
            f"融合 P10 / P50 / P90：{result.p10:,} / {result.p50:,} / {result.p90:,}"
        )

    if result.minimum is not None and result.maximum is not None:
        lines.append(f"兼容世界范围：{result.minimum:,} - {result.maximum:,}")
    if result.recommended_bid is not None:
        lines.append(
            f"建议最高出价（仅人工参考）：{result.recommended_bid:,} "
            f"= P10 × {result.bid_safety_factor:.0%}"
        )
    if (
        result.required_winning_multiplier is not None
        and result.maximum_opponent_bid is not None
    ):
        lines.append(
            f"本轮中标倍率 ×{result.required_winning_multiplier:g}；"
            f"对应对手最高报价约 {result.maximum_opponent_bid:,}"
        )
    if diagnostics:
        label = "v2 诊断" if result.estimate_source == "v6_default" else "冲突定位"
        lines.append(f"{label}：{diagnostics}")
    if issues:
        lines.append(f"提示：{issues}")
    lines.append("不会自动提交报价；区间未校准，actionable=false")
    return "\n".join(lines)
