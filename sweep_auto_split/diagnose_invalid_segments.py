"""
诊断无效 segment 的原因

分析为什么某些 sweep 没有生成有效的 segment

使用方法:
    python -m sweep_auto_split.diagnose_invalid_segments --input /path/to/dataset
"""

from typing import List, Tuple, Dict
from .config import SweepSegmentConfig, SweepKeypoint, SegmentBoundary


def diagnose_invalid_segment(
    kp: SweepKeypoint,
    t: int,
    valid_keypoints: List[SweepKeypoint],
    episode_length: int,
    config: SweepSegmentConfig
) -> Tuple[bool, str, dict]:
    """
    诊断单个 segment 无效的原因

    Returns:
        (is_valid, reason, details)
    """
    H = config.H
    A_min = config.A_min
    R_min = config.R_min

    # 相邻 sweep
    P_prev_1 = valid_keypoints[t - 1].P_t1 if t > 0 else -1
    P_next_0 = valid_keypoints[t + 1].P_t0 if t < len(valid_keypoints) - 1 else episode_length + H

    P_t0 = kp.P_t0
    P_t1 = kp.P_t1

    # 四个约束
    constraint_2 = P_t1 - H + 1 + R_min  # s >= 这个值
    constraint_3 = P_prev_1 + 1           # s >= 这个值
    constraint_1 = P_t0 - A_min           # s <= 这个值
    constraint_4 = P_next_0 - H           # s <= 这个值

    s_min = max(constraint_3, constraint_2)
    s_max = min(constraint_1, constraint_4)

    is_valid = s_min <= s_max

    details = {
        "P_t0": P_t0,
        "P_t1": P_t1,
        "P_prev_1": P_prev_1,
        "P_next_0": P_next_0,
        "constraint_1": constraint_1,
        "constraint_2": constraint_2,
        "constraint_3": constraint_3,
        "constraint_4": constraint_4,
        "s_min": s_min,
        "s_max": s_max,
        "gap_to_prev": P_t0 - P_prev_1 if t > 0 else None,
        "gap_to_next": P_next_0 - P_t1 if t < len(valid_keypoints) - 1 else None,
    }

    if is_valid:
        return True, "Valid", details

    # 分析无效原因
    reasons = []

    # 检查哪个约束导致了问题
    if constraint_3 > constraint_1:
        reasons.append(f"约束3>约束1: 上一个sweep结束太晚 (P_prev_1={P_prev_1}) 导致 s_min={constraint_3} > s_max={constraint_1}")

    if constraint_3 > constraint_4:
        gap = P_next_0 - P_prev_1
        reasons.append(f"约束3>约束4: 与下一个sweep间隔太小 (间隔={gap}帧 < H={H})")

    if constraint_2 > constraint_1:
        sweep_len = P_t1 - P_t0 + 1
        max_len = H - A_min - R_min
        reasons.append(f"约束2>约束1: sweep太长 (L={sweep_len} > H-A_min-R_min={max_len})")

    if constraint_2 > constraint_4:
        gap_to_next = P_next_0 - P_t1
        reasons.append(f"约束2>约束4: 与下一个sweep间隔太小 (间隔={gap_to_next}帧 < H-R_min={H-R_min})")

    if not reasons:
        reasons.append(f"s_min={s_min} > s_max={s_max}")

    return False, "; ".join(reasons), details


def analyze_invalid_reasons(
    all_results: Dict[int, Tuple[List[SweepKeypoint], List[SegmentBoundary]]],
    episode_lengths: Dict[int, int],
    config: SweepSegmentConfig
) -> Dict[str, int]:
    """
    分析所有无效 segment 的原因统计

    Returns:
        原因 -> 数量 的字典
    """
    reason_counts = {
        "与下一个sweep间隔太小": 0,
        "与上一个sweep间隔太小": 0,
        "sweep本身太长": 0,
        "其他": 0,
    }

    for ep_id, (keypoints, boundaries) in all_results.items():
        valid_keypoints = [kp for kp in keypoints if kp.is_valid]
        episode_length = episode_lengths.get(ep_id, 1000)

        for t, kp in enumerate(valid_keypoints):
            if t < len(boundaries) and boundaries[t].is_valid:
                continue

            is_valid, reason, details = diagnose_invalid_segment(
                kp, t, valid_keypoints, episode_length, config
            )

            if not is_valid:
                if "约束4" in reason or "下一个sweep间隔太小" in reason:
                    reason_counts["与下一个sweep间隔太小"] += 1
                elif "约束3" in reason or "上一个sweep" in reason:
                    reason_counts["与上一个sweep间隔太小"] += 1
                elif "太长" in reason:
                    reason_counts["sweep本身太长"] += 1
                else:
                    reason_counts["其他"] += 1

    return reason_counts


def print_invalid_segments_report(
    keypoints: List[SweepKeypoint],
    boundaries: List[SegmentBoundary],
    episode_length: int,
    config: SweepSegmentConfig,
    episode_id: int = 0,
    max_show: int = 10,
):
    """
    打印无效 segment 的详细报告
    """
    valid_keypoints = [kp for kp in keypoints if kp.is_valid]

    print(f"\n[Episode {episode_id}] Invalid Segments Diagnosis")
    print("-" * 60)

    invalid_count = 0
    reason_counts = {}

    for t, kp in enumerate(valid_keypoints):
        boundary = boundaries[t] if t < len(boundaries) else None

        if boundary and boundary.is_valid:
            continue

        is_valid, reason, details = diagnose_invalid_segment(
            kp, t, valid_keypoints, episode_length, config
        )

        if not is_valid:
            invalid_count += 1

            # 统计原因
            for r in reason.split("; "):
                key = r.split(":")[0] if ":" in r else r
                reason_counts[key] = reason_counts.get(key, 0) + 1

            if invalid_count <= max_show:
                print(f"\n  [Sweep {kp.sweep_idx}] (t={t})")
                print(f"    P_t0={details['P_t0']}, P_t1={details['P_t1']}")
                if details['gap_to_prev'] is not None:
                    print(f"    与上一个sweep间隔: {details['gap_to_prev']} 帧")
                if details['gap_to_next'] is not None:
                    print(f"    与下一个sweep间隔: {details['gap_to_next']} 帧")
                print(f"    s_min={details['s_min']}, s_max={details['s_max']}")
                print(f"    原因: {reason}")

    if invalid_count > max_show:
        print(f"\n  ... 还有 {invalid_count - max_show} 个无效 segment")

    print(f"\n  共 {invalid_count} 个无效 segment")
    if reason_counts:
        print(f"  原因分布:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")


def main():
    """命令行入口"""
    import argparse
    from .data_loader import LeRobotDataLoader
    from .sweep_detector import SweepDetector
    from .segment_calculator import SegmentCalculator

    parser = argparse.ArgumentParser(
        description="诊断无效 segment 的原因"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="LeRobot 数据集路径"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=5,
        help="最大处理 episode 数 (default: 5)"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=30,
        help="action horizon (default: 30)"
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="left",
        choices=["left", "right", "both"],
        help="使用哪只手臂 (default: left)"
    )

    args = parser.parse_args()

    # 配置
    config = SweepSegmentConfig(
        H=args.H,
        active_arm=args.arm,
        verbose=False,
    )

    # 加载数据
    data_loader = LeRobotDataLoader(args.input)
    detector = SweepDetector(config)
    calculator = SegmentCalculator(config)

    print("=" * 70)
    print("INVALID SEGMENTS DIAGNOSIS")
    print("=" * 70)
    print(f"Dataset: {args.input}")
    print(f"H={config.H}, A_min={config.A_min}, R_min={config.R_min}")

    # 收集所有结果
    all_results = {}
    episode_lengths = {}
    total_valid = 0
    total_invalid = 0

    episode_ids = data_loader.get_episode_list()[:args.max_episodes]

    for ep_id in episode_ids:
        episode_data = data_loader.load_episode(ep_id)
        keypoints = detector.detect_keypoints(
            episode_data.state_trajectory,
            episode_data.ee_pose_trajectory
        )
        boundaries = calculator.calculate_boundaries(keypoints, episode_data.length)

        all_results[ep_id] = (keypoints, boundaries)
        episode_lengths[ep_id] = episode_data.length

        valid = sum(1 for b in boundaries if b.is_valid)
        invalid = len(boundaries) - valid
        total_valid += valid
        total_invalid += invalid

        # 打印这个 episode 的无效 segment
        if invalid > 0:
            print_invalid_segments_report(
                keypoints, boundaries, episode_data.length, config,
                episode_id=ep_id, max_show=3
            )

    # 总体统计
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total segments: {total_valid + total_invalid}")
    print(f"Valid: {total_valid} ({total_valid/(total_valid+total_invalid)*100:.1f}%)")
    print(f"Invalid: {total_invalid} ({total_invalid/(total_valid+total_invalid)*100:.1f}%)")

    # 原因统计
    reason_stats = analyze_invalid_reasons(all_results, episode_lengths, config)
    print(f"\n无效原因分布:")
    for reason, count in sorted(reason_stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {reason}: {count} ({count/max(1,total_invalid)*100:.1f}%)")

    print("=" * 70)


if __name__ == "__main__":
    main()
