"""
Segment 有效性统计与诊断模块

目标：对每个 episode、每个 sweep 统计并解释为什么 segment 会被判为无效，
     给出"主导约束"和"可行性指标"，用于快速定位问题来源。

输出：
    - sweep_segment_stats.csv: 每行一个 sweep 的详细统计
    - summary.json: 全局汇总 + per-episode 汇总

使用方法:
    python -m sweep_auto_split.segment_stats --input /path/to/dataset -o ./output
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import Counter

from .config import SweepSegmentConfig, SweepKeypoint


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SweepStats:
    """单个 sweep 的统计信息"""
    # 基础信息
    episode_id: int
    sweep_order: int           # 该 episode 内第几个有效 sweep (0,1,2...)
    sweep_idx: int             # 原始编号
    P_t0: int
    P_t1: int
    episode_length: int
    H: int
    A_min: int
    R_min: int

    # 四个约束值
    constraint_1: int          # P_t0 - A_min
    constraint_2: int          # P_t1 - H + 1 + R_min
    constraint_3: int          # P_prev_1 + 1
    constraint_4: int          # P_next_0 - H
    P_prev_1: int              # 上一个 sweep 的结束帧 (-1 if first)
    P_next_0: int              # 下一个 sweep 的开始帧 (episode_length+H if last)

    # 起点可行区间（含 episode 边界约束）
    s_min: int
    s_max: int
    s_min_raw: int             # 截断前的 s_min
    s_max_raw: int             # 截断前的 s_max

    # 有效性与主导原因
    is_valid: bool
    dominant_lower: str        # 把 s_min 顶上去的约束
    dominant_upper: str        # 把 s_max 压下来的约束
    diversity: int

    # 可行性指标
    D: int                     # sweep 时长 = P_t1 - P_t0
    M: int                     # 窗口可容纳上限 = H - 1 - (A_min + R_min)
    feasible_by_duration: bool # D <= M

    # 额外信息
    gap_to_prev: Optional[int] = None   # 与上一个 sweep 的间隔
    gap_to_next: Optional[int] = None   # 与下一个 sweep 的间隔
    is_first_sweep: bool = False
    is_last_sweep: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EpisodeSummary:
    """单个 episode 的汇总"""
    episode_id: int
    episode_length: int
    num_sweeps_valid_kp: int           # 有效关键点数
    num_segments_valid: int            # 有效 segment 数
    valid_rate: float
    first_sweep_is_valid: bool
    first_sweep_invalid_reason: Optional[str] = None

    # 无效原因计数
    invalid_reason_counts: Dict[str, int] = field(default_factory=dict)

    # D > M 的数量
    duration_exceeds_margin: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GlobalSummary:
    """全局汇总"""
    total_episodes: int
    total_sweeps: int
    total_valid_segments: int
    overall_valid_rate: float

    # 第一个 sweep 统计
    first_sweep_total: int
    first_sweep_valid: int
    first_sweep_invalid_rate: float

    # D > M 统计
    duration_exceeds_margin_total: int
    duration_exceeds_margin_first_sweep: int
    duration_exceeds_margin_rate: float

    # dominant 分布 Top-K
    dominant_lower_distribution: Dict[str, int] = field(default_factory=dict)
    dominant_upper_distribution: Dict[str, int] = field(default_factory=dict)
    invalid_reason_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# 核心计算
# ============================================================

def compute_sweep_stats(
    kp: SweepKeypoint,
    sweep_order: int,
    valid_keypoints: List[SweepKeypoint],
    episode_id: int,
    episode_length: int,
    config: SweepSegmentConfig
) -> SweepStats:
    """
    计算单个 sweep 的详细统计信息

    Args:
        kp: 当前 sweep 的关键点
        sweep_order: 在 episode 内的顺序 (0, 1, 2, ...)
        valid_keypoints: 所有有效关键点列表
        episode_id: episode ID
        episode_length: episode 总帧数
        config: 配置

    Returns:
        SweepStats 对象
    """
    H = config.H
    A_min = config.A_min
    R_min = config.R_min
    t = sweep_order

    # 相邻 sweep 信息
    P_prev_1 = valid_keypoints[t - 1].P_t1 if t > 0 else -1
    P_next_0 = valid_keypoints[t + 1].P_t0 if t < len(valid_keypoints) - 1 else episode_length + H

    P_t0 = kp.P_t0
    P_t1 = kp.P_t1

    # 四个约束
    constraint_1 = P_t0 - A_min
    constraint_2 = P_t1 - H + 1 + R_min
    constraint_3 = P_prev_1 + 1
    constraint_4 = P_next_0 - H

    # 原始 s_min, s_max（不含 episode 边界约束）
    s_min_raw = max(constraint_2, constraint_3)
    s_max_raw = min(constraint_1, constraint_4)

    # 加上 episode 边界约束
    episode_bound_lower = 0
    episode_bound_upper = episode_length - H

    s_min = max(constraint_2, constraint_3, episode_bound_lower)
    s_max = min(constraint_1, constraint_4, episode_bound_upper)

    # 有效性
    is_valid = s_min <= s_max

    # 主导约束分析
    lower_candidates = {
        "constraint_2": constraint_2,
        "constraint_3": constraint_3,
        "episode_start": episode_bound_lower,
    }
    upper_candidates = {
        "constraint_1": constraint_1,
        "constraint_4": constraint_4,
        "episode_end": episode_bound_upper,
    }

    dominant_lower = max(lower_candidates, key=lower_candidates.get)
    dominant_upper = min(upper_candidates, key=upper_candidates.get)

    # 多样性
    diversity = max(0, s_max - s_min + 1) if is_valid else 0

    # 可行性指标
    D = P_t1 - P_t0                    # sweep 时长
    M = H - 1 - (A_min + R_min)        # 窗口可容纳上限
    feasible_by_duration = (D <= M)

    # 间隔计算
    gap_to_prev = P_t0 - P_prev_1 if t > 0 else None
    gap_to_next = P_next_0 - P_t1 if t < len(valid_keypoints) - 1 else None

    return SweepStats(
        episode_id=episode_id,
        sweep_order=sweep_order,
        sweep_idx=kp.sweep_idx,
        P_t0=P_t0,
        P_t1=P_t1,
        episode_length=episode_length,
        H=H,
        A_min=A_min,
        R_min=R_min,
        constraint_1=constraint_1,
        constraint_2=constraint_2,
        constraint_3=constraint_3,
        constraint_4=constraint_4,
        P_prev_1=P_prev_1,
        P_next_0=P_next_0,
        s_min=s_min,
        s_max=s_max,
        s_min_raw=s_min_raw,
        s_max_raw=s_max_raw,
        is_valid=is_valid,
        dominant_lower=dominant_lower,
        dominant_upper=dominant_upper,
        diversity=diversity,
        D=D,
        M=M,
        feasible_by_duration=feasible_by_duration,
        gap_to_prev=gap_to_prev,
        gap_to_next=gap_to_next,
        is_first_sweep=(t == 0),
        is_last_sweep=(t == len(valid_keypoints) - 1),
    )


def compute_episode_summary(
    episode_id: int,
    episode_length: int,
    sweep_stats_list: List[SweepStats]
) -> EpisodeSummary:
    """
    计算单个 episode 的汇总统计

    Args:
        episode_id: episode ID
        episode_length: episode 总帧数
        sweep_stats_list: 该 episode 所有 sweep 的统计

    Returns:
        EpisodeSummary 对象
    """
    num_sweeps = len(sweep_stats_list)
    num_valid = sum(1 for s in sweep_stats_list if s.is_valid)
    valid_rate = num_valid / num_sweeps if num_sweeps > 0 else 0.0

    # 第一个 sweep
    first_sweep_is_valid = sweep_stats_list[0].is_valid if sweep_stats_list else True
    first_sweep_invalid_reason = None
    if sweep_stats_list and not first_sweep_is_valid:
        s = sweep_stats_list[0]
        first_sweep_invalid_reason = f"lower={s.dominant_lower}, upper={s.dominant_upper}"

    # 无效原因计数
    invalid_reason_counts = Counter()
    for s in sweep_stats_list:
        if not s.is_valid:
            reason = f"lower={s.dominant_lower} & upper={s.dominant_upper}"
            invalid_reason_counts[reason] += 1

    # D > M 计数
    duration_exceeds = sum(1 for s in sweep_stats_list if not s.feasible_by_duration)

    return EpisodeSummary(
        episode_id=episode_id,
        episode_length=episode_length,
        num_sweeps_valid_kp=num_sweeps,
        num_segments_valid=num_valid,
        valid_rate=valid_rate,
        first_sweep_is_valid=first_sweep_is_valid,
        first_sweep_invalid_reason=first_sweep_invalid_reason,
        invalid_reason_counts=dict(invalid_reason_counts),
        duration_exceeds_margin=duration_exceeds,
    )


def compute_global_summary(
    episode_summaries: List[EpisodeSummary],
    all_sweep_stats: List[SweepStats]
) -> GlobalSummary:
    """
    计算全局汇总统计

    Args:
        episode_summaries: 所有 episode 的汇总
        all_sweep_stats: 所有 sweep 的统计

    Returns:
        GlobalSummary 对象
    """
    total_episodes = len(episode_summaries)
    total_sweeps = len(all_sweep_stats)
    total_valid = sum(1 for s in all_sweep_stats if s.is_valid)
    overall_valid_rate = total_valid / total_sweeps if total_sweeps > 0 else 0.0

    # 第一个 sweep 统计
    first_sweeps = [s for s in all_sweep_stats if s.is_first_sweep]
    first_sweep_total = len(first_sweeps)
    first_sweep_valid = sum(1 for s in first_sweeps if s.is_valid)
    first_sweep_invalid_rate = (first_sweep_total - first_sweep_valid) / first_sweep_total if first_sweep_total > 0 else 0.0

    # D > M 统计
    duration_exceeds_total = sum(1 for s in all_sweep_stats if not s.feasible_by_duration)
    duration_exceeds_first = sum(1 for s in first_sweeps if not s.feasible_by_duration)
    duration_exceeds_rate = duration_exceeds_total / total_sweeps if total_sweeps > 0 else 0.0

    # dominant 分布
    dominant_lower_counts = Counter(s.dominant_lower for s in all_sweep_stats if not s.is_valid)
    dominant_upper_counts = Counter(s.dominant_upper for s in all_sweep_stats if not s.is_valid)
    invalid_reason_counts = Counter()
    for s in all_sweep_stats:
        if not s.is_valid:
            reason = f"lower={s.dominant_lower} & upper={s.dominant_upper}"
            invalid_reason_counts[reason] += 1

    return GlobalSummary(
        total_episodes=total_episodes,
        total_sweeps=total_sweeps,
        total_valid_segments=total_valid,
        overall_valid_rate=overall_valid_rate,
        first_sweep_total=first_sweep_total,
        first_sweep_valid=first_sweep_valid,
        first_sweep_invalid_rate=first_sweep_invalid_rate,
        duration_exceeds_margin_total=duration_exceeds_total,
        duration_exceeds_margin_first_sweep=duration_exceeds_first,
        duration_exceeds_margin_rate=duration_exceeds_rate,
        dominant_lower_distribution=dict(dominant_lower_counts.most_common()),
        dominant_upper_distribution=dict(dominant_upper_counts.most_common()),
        invalid_reason_distribution=dict(invalid_reason_counts.most_common()),
    )


# ============================================================
# 输出函数
# ============================================================

def save_sweep_stats_csv(sweep_stats: List[SweepStats], output_path: Path):
    """保存 sweep 统计到 CSV"""
    if not sweep_stats:
        return

    fieldnames = list(sweep_stats[0].to_dict().keys())

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stat in sweep_stats:
            writer.writerow(stat.to_dict())


def save_summary_json(
    global_summary: GlobalSummary,
    episode_summaries: List[EpisodeSummary],
    output_path: Path
):
    """保存汇总到 JSON"""
    data = {
        "global": global_summary.to_dict(),
        "episodes": [ep.to_dict() for ep in episode_summaries],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_summary_report(
    global_summary: GlobalSummary,
    episode_summaries: List[EpisodeSummary],
    all_sweep_stats: List[SweepStats],
    top_n: int = 5
):
    """打印终端简报"""
    print("\n" + "=" * 70)
    print("SEGMENT VALIDITY STATISTICS REPORT")
    print("=" * 70)

    # 全局统计
    gs = global_summary
    print(f"\n【全局统计】")
    print(f"  总 Episode 数: {gs.total_episodes}")
    print(f"  总 Sweep 数: {gs.total_sweeps}")
    print(f"  有效 Segment 数: {gs.total_valid_segments}")
    print(f"  总体有效率: {gs.overall_valid_rate * 100:.1f}%")

    print(f"\n【第一个 Sweep 统计】")
    print(f"  总数: {gs.first_sweep_total}")
    print(f"  有效: {gs.first_sweep_valid}")
    print(f"  无效率: {gs.first_sweep_invalid_rate * 100:.1f}%")

    print(f"\n【D > M 统计】(sweep时长超过窗口可容纳上限)")
    print(f"  总数: {gs.duration_exceeds_margin_total} ({gs.duration_exceeds_margin_rate * 100:.1f}%)")
    print(f"  首个 Sweep: {gs.duration_exceeds_margin_first_sweep}")

    # 无效原因分布
    if gs.invalid_reason_distribution:
        print(f"\n【无效原因分布 Top-{top_n}】")
        for i, (reason, count) in enumerate(list(gs.invalid_reason_distribution.items())[:top_n]):
            pct = count / (gs.total_sweeps - gs.total_valid_segments) * 100 if gs.total_sweeps > gs.total_valid_segments else 0
            print(f"  {i+1}. {reason}: {count} ({pct:.1f}%)")

    # 主导约束分布
    if gs.dominant_lower_distribution:
        print(f"\n【主导下界约束分布】(把 s_min 顶上去的)")
        for constraint, count in gs.dominant_lower_distribution.items():
            print(f"  {constraint}: {count}")

    if gs.dominant_upper_distribution:
        print(f"\n【主导上界约束分布】(把 s_max 压下来的)")
        for constraint, count in gs.dominant_upper_distribution.items():
            print(f"  {constraint}: {count}")

    # 有效率最低的 episode
    sorted_episodes = sorted(episode_summaries, key=lambda x: x.valid_rate)
    print(f"\n【有效率最低的 {min(top_n, len(sorted_episodes))} 个 Episode】")
    for ep in sorted_episodes[:top_n]:
        print(f"  Episode {ep.episode_id}: {ep.valid_rate * 100:.1f}% "
              f"({ep.num_segments_valid}/{ep.num_sweeps_valid_kp})")
        if ep.first_sweep_invalid_reason:
            print(f"    首个 Sweep 无效原因: {ep.first_sweep_invalid_reason}")

    # 首个 sweep 失败原因样例
    first_sweep_invalid = [s for s in all_sweep_stats if s.is_first_sweep and not s.is_valid]
    if first_sweep_invalid:
        print(f"\n【首个 Sweep 失败样例】(共 {len(first_sweep_invalid)} 个)")
        # 按原因分组
        reason_examples = {}
        for s in first_sweep_invalid:
            reason = f"lower={s.dominant_lower} & upper={s.dominant_upper}"
            if reason not in reason_examples:
                reason_examples[reason] = []
            if len(reason_examples[reason]) < 2:  # 每种原因最多 2 个样例
                reason_examples[reason].append(s)

        for reason, examples in list(reason_examples.items())[:3]:  # 最多 3 种原因
            print(f"\n  原因: {reason}")
            for s in examples:
                print(f"    Episode {s.episode_id}: P=[{s.P_t0}, {s.P_t1}], "
                      f"s_min={s.s_min}, s_max={s.s_max}, "
                      f"gap_to_next={s.gap_to_next}, D={s.D}, M={s.M}")

    print("\n" + "=" * 70)


# ============================================================
# 主处理函数
# ============================================================

def analyze_segment_validity(
    results: Dict[int, Tuple[List[SweepKeypoint], Any]],
    episode_lengths: Dict[int, int],
    config: SweepSegmentConfig,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[GlobalSummary, List[EpisodeSummary], List[SweepStats]]:
    """
    分析所有 episode 的 segment 有效性

    Args:
        results: {episode_id: (keypoints, boundaries)} 的字典
        episode_lengths: {episode_id: length} 的字典
        config: 配置
        output_dir: 输出目录（可选）
        verbose: 是否打印详细信息

    Returns:
        (global_summary, episode_summaries, all_sweep_stats)
    """
    all_sweep_stats = []
    episode_summaries = []

    for episode_id, (keypoints, _) in results.items():
        valid_keypoints = [kp for kp in keypoints if kp.is_valid]
        episode_length = episode_lengths.get(episode_id, 1000)

        episode_stats = []
        for t, kp in enumerate(valid_keypoints):
            stats = compute_sweep_stats(
                kp, t, valid_keypoints, episode_id, episode_length, config
            )
            episode_stats.append(stats)
            all_sweep_stats.append(stats)

        if episode_stats:
            ep_summary = compute_episode_summary(episode_id, episode_length, episode_stats)
            episode_summaries.append(ep_summary)

    global_summary = compute_global_summary(episode_summaries, all_sweep_stats)

    # 保存输出
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_sweep_stats_csv(all_sweep_stats, output_path / "sweep_segment_stats.csv")
        save_summary_json(global_summary, episode_summaries, output_path / "summary.json")

        if verbose:
            print(f"\n输出已保存到:")
            print(f"  - {output_path / 'sweep_segment_stats.csv'}")
            print(f"  - {output_path / 'summary.json'}")

    # 打印简报
    if verbose:
        print_summary_report(global_summary, episode_summaries, all_sweep_stats)

    return global_summary, episode_summaries, all_sweep_stats


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行入口"""
    import argparse
    from .data_loader import LeRobotDataLoader
    from .sweep_detector import SweepDetector
    from .segment_calculator import SegmentCalculator

    parser = argparse.ArgumentParser(
        description="Segment 有效性统计与诊断"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="LeRobot 数据集路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./segment_stats_output",
        help="输出目录 (default: ./segment_stats_output)"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最大处理 episode 数"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=30,
        help="action horizon (default: 30)"
    )
    parser.add_argument(
        "--A-min",
        type=int,
        default=2,
        help="Approach 最少帧数 (default: 2)"
    )
    parser.add_argument(
        "--R-min",
        type=int,
        default=2,
        help="Retreat 最少帧数 (default: 2)"
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="left",
        choices=["left", "right", "both"],
        help="使用哪只手臂 (default: left)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式，不打印详细信息"
    )

    args = parser.parse_args()

    # 配置
    config = SweepSegmentConfig(
        H=args.H,
        A_min=args.A_min,
        R_min=args.R_min,
        active_arm=args.arm,
        verbose=False,
    )

    # 加载数据
    print(f"加载数据集: {args.input}")
    data_loader = LeRobotDataLoader(args.input)
    detector = SweepDetector(config)
    calculator = SegmentCalculator(config)

    # 处理每个 episode
    results = {}
    episode_lengths = {}

    episode_ids = data_loader.get_episode_list()
    if args.max_episodes:
        episode_ids = episode_ids[:args.max_episodes]

    print(f"处理 {len(episode_ids)} 个 episode...")

    for ep_id in episode_ids:
        episode_data = data_loader.load_episode(ep_id)
        keypoints = detector.detect_keypoints(
            episode_data.state_trajectory,
            episode_data.ee_pose_trajectory
        )
        boundaries = calculator.calculate_boundaries(keypoints, episode_data.length)

        results[ep_id] = (keypoints, boundaries)
        episode_lengths[ep_id] = episode_data.length

    # 分析
    analyze_segment_validity(
        results=results,
        episode_lengths=episode_lengths,
        config=config,
        output_dir=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
