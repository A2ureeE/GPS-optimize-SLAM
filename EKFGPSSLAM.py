# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from pyproj import Proj
from pyproj.exceptions import CRSError
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, messagebox # 导入 messagebox
import traceback
from typing import Dict, Tuple, Optional, List, Any

# ----------------------------
# 配置参数 (修改 GPS RANSAC 部分)
# ----------------------------
CONFIG = {
    # EKF 参数
    "ekf": {
        "initial_cov_diag": [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01],
        "process_noise_diag": [1, 1, 1, 0.1, 0.1, 0.1, 0.1],
        "meas_noise_diag": [0.1, 0.1, 0.1],
    },
    # Sim(3) 全局变换 RANSAC 参数
    "sim3_ransac": {
        "min_samples": 4,
        "residual_threshold": 0.1,
        "max_trials": 1000,
        "min_inliers_for_transform": 4,
    },
    # === 修改: GPS 轨迹 RANSAC 滤波参数 (增加滑动窗口选项) ===
    "gps_filtering_ransac": {
        "enabled": True, # 是否启用 GPS RANSAC 滤波
        "use_sliding_window": True,  # <<< 新增: True 使用滑动窗口, False 使用全局 RANSAC
        # --- 滑动窗口参数 (仅当 use_sliding_window 为 True 时有效) ---
        "window_duration_seconds": 15.0, # <<< 新增: 滑动窗口的时间长度 (秒) - 需要调优
        "window_step_factor": 0.5,    # <<< 新增: 窗口移动步长占窗口时长的比例 (0.5 表示 50% 重叠) - 需要调优
        # ------------------------------------------------------
        "polynomial_degree": 2, # 多项式阶数 (对于局部拟合，2或3通常足够)
        "min_samples": 6,       # RANSAC 最小样本数 (应 >= degree + 1)，现在应用于 *每个窗口*
        "residual_threshold_meters": 10.0, # 内点阈值 (米)，现在是相对于 *局部* 窗口拟合的偏差 - *需要根据GPS精度仔细调优*
        "max_trials": 50,       # RANSAC 最大迭代次数 (每个窗口的迭代次数，可以适当减少)
    },
    # 时间对齐参数
    "time_alignment": {
        "max_samples_for_corr": 500,
        "max_gps_gap_threshold": 5.0, # GPS 信号中断阈值 (秒)
    }
}

# ----------------------------
# 增强数据加载与预处理
# ----------------------------

def load_slam_trajectory(txt_path: str) -> Dict[str, np.ndarray]:
    """加载并验证SLAM轨迹数据 (TUM格式)"""
    try:
        data = np.loadtxt(txt_path)
        if data.ndim == 1:
             data = data.reshape(1, -1)
        assert data.shape[1] == 8, f"SLAM文件格式错误：需要8列 (ts x y z qx qy qz qw), 找到 {data.shape[1]} 列"
        return {
            'timestamps': data[:, 0].astype(float),
            'positions': data[:, 1:4].astype(float),
            'quaternions': data[:, 4:8].astype(float)
        }
    except FileNotFoundError:
        raise ValueError(f"SLAM文件未找到: {txt_path}")
    except Exception as e:
        raise ValueError(f"SLAM数据加载或解析失败 ({txt_path}): {str(e)}")

def auto_utm_projection(lons: np.ndarray, lats: np.ndarray) -> Tuple[int, str]:
    """根据经度自动计算UTM分区号和南北半球"""
    if lons.size == 0 or lats.size == 0:
        raise ValueError("经纬度数据不能为空以确定UTM分区")
    central_lon = np.mean(lons)
    zone = int((central_lon + 180) // 6 + 1)
    hemisphere = ' +south' if np.mean(lats) < 0 else ''
    return zone, hemisphere

# -------------------------------------------
# GPS 离群点过滤模块 (修改后支持滑动窗口)
# -------------------------------------------
def filter_gps_outliers_ransac(times: np.ndarray, positions: np.ndarray,
                               config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 RANSAC 和多项式模型过滤 GPS 轨迹中的离群点。
    支持全局拟合或滑动窗口局部拟合。

    Args:
        times: GPS 时间戳数组 (N,)
        positions: GPS 位置数组 (N, 3) - 期望是 UTM 坐标
        config: 包含 RANSAC 参数的字典 (来自 CONFIG['gps_filtering_ransac'])

    Returns:
        Tuple[np.ndarray, np.ndarray]: 过滤后的时间戳和位置数组 (M,), (M, 3) where M <= N
    """
    if not config.get("enabled", False):
        print("GPS RANSAC 过滤被禁用。")
        return times, positions

    n_points = len(times)
    min_samples_needed = config['min_samples'] # 全局或每个窗口所需的最小样本数

    if n_points < min_samples_needed:
        print(f"警告: GPS 点数 ({n_points}) 少于 RANSAC 最小样本数 ({min_samples_needed})，跳过 GPS 离群点过滤。")
        return times, positions

    use_sliding_window = config.get("use_sliding_window", False)

    if not use_sliding_window:
        # --- 执行原始的全局 RANSAC 拟合 ---
        print("正在执行全局 GPS RANSAC 过滤...")
        try:
            t_feature = times.reshape(-1, 1)
            inlier_masks = []
            for i in range(positions.shape[1]): # 对 X, Y, Z 分别拟合
                target = positions[:, i]
                model = make_pipeline(
                    PolynomialFeatures(degree=config['polynomial_degree']),
                    RANSACRegressor(
                        min_samples=min_samples_needed,
                        residual_threshold=config['residual_threshold_meters'],
                        max_trials=config['max_trials'],
                    )
                )
                model.fit(t_feature, target)
                inlier_mask_dim = model[-1].inlier_mask_
                inlier_masks.append(inlier_mask_dim)

            final_inlier_mask = np.logical_and.reduce(inlier_masks) # 要求 X,Y,Z 都是内点
            num_inliers = np.sum(final_inlier_mask)
            num_outliers = n_points - num_inliers

            if num_outliers > 0:
                print(f"  全局 RANSAC: 识别并移除了 {num_outliers} 个离群点 (保留 {num_inliers} / {n_points} 个点)。")
            else:
                print("  全局 RANSAC: 未发现离群点。")

            if num_inliers < min_samples_needed:
                 print(f"警告: 全局 RANSAC 过滤后剩余的 GPS 点数 ({num_inliers}) 过少。")

            return times[final_inlier_mask], positions[final_inlier_mask]

        except Exception as e:
            print(f"全局 GPS RANSAC 过滤过程中发生错误: {e}. 跳过过滤步骤。")
            traceback.print_exc()
            return times, positions
        # --- 全局 RANSAC 结束 ---

    else:
        # --- 执行滑动窗口 RANSAC 拟合 ---
        window_duration = config['window_duration_seconds']
        step_factor = config['window_step_factor']
        window_step = window_duration * step_factor
        residual_threshold = config['residual_threshold_meters']
        poly_degree = config['polynomial_degree']
        max_trials_per_window = config['max_trials'] # 每个窗口的 max_trials

        print(f"正在执行滑动窗口 GPS RANSAC 过滤 (窗口时长: {window_duration}s, 步长: {window_step:.2f}s)...")

        if n_points < min_samples_needed:
             print(f"警告：总点数 ({n_points}) 少于窗口最小样本数 ({min_samples_needed})，无法使用滑动窗口。")
             return times, positions # 无法处理，返回原数据

        # 初始化一个全局内点掩码，初始全为 False
        # 一个点最终要保留，必须至少在一个窗口中被判断为内点
        overall_inlier_mask = np.zeros(n_points, dtype=bool)
        processed_windows = 0
        successful_windows = 0

        start_time = times[0]
        end_time = times[-1]
        current_window_start = start_time

        while current_window_start < end_time:
            current_window_end = current_window_start + window_duration

            # 查找落入当前窗口的时间戳索引 (包含起始，不包含结束，以避免步长为0时重复处理)
            window_indices = np.where((times >= current_window_start) & (times < current_window_end))[0]
            n_window_points = len(window_indices)

            # 确保窗口内有足够的点进行 RANSAC
            if n_window_points >= min_samples_needed:
                processed_windows += 1
                # 提取当前窗口的数据
                window_times = times[window_indices]
                window_positions = positions[window_indices]
                window_t_feature = window_times.reshape(-1, 1)

                try:
                    window_inlier_masks_dim = [] # 存储当前窗口内各维度的内点掩码
                    valid_window_fit = True
                    for i in range(positions.shape[1]): # 对 X, Y, Z 维度分别拟合
                        target = window_positions[:, i]
                        model = make_pipeline(
                            PolynomialFeatures(degree=poly_degree),
                            RANSACRegressor(
                                min_samples=min_samples_needed, # 窗口内的最小样本
                                residual_threshold=residual_threshold, # 局部阈值
                                max_trials=max_trials_per_window, # 窗口内的迭代次数
                            )
                        )
                        model.fit(window_t_feature, target)
                        # 获取相对于当前窗口的内点掩码
                        inlier_mask_window_dim = model[-1].inlier_mask_
                        window_inlier_masks_dim.append(inlier_mask_window_dim)

                        # (可选) 如果某个维度内点过少，可以提前判定此窗口拟合失败
                        # if np.sum(inlier_mask_window_dim) < min_samples_needed:
                        #    valid_window_fit = False
                        #    break

                    if valid_window_fit:
                        # 合并当前窗口的 X, Y, Z 维度的内点掩码 (要求在局部窗口内 X,Y,Z 都是内点)
                        window_final_inlier_mask = np.logical_and.reduce(window_inlier_masks_dim)
                        # 获取这些内点在 *原始* 数组中的索引
                        original_indices_of_inliers = window_indices[window_final_inlier_mask]
                        # 将这些点在全局掩码中标记为 True (只要在一个窗口是内点，就标记)
                        overall_inlier_mask[original_indices_of_inliers] = True
                        successful_windows += 1
                    # else:
                        # print(f"  窗口 [{current_window_start:.2f}s - {current_window_end:.2f}s] RANSAC 失败 (内点不足)")
                        # pass

                except Exception as e:
                     # 捕获单个窗口内的 RANSAC 错误
                     print(f"警告：窗口 [{current_window_start:.2f}s - {current_window_end:.2f}s] RANSAC 拟合失败: {e}")
                     # traceback.print_exc() # 可选的详细错误
                     # 继续处理下一个窗口
            # else:
                 # print(f"  跳过窗口 [{current_window_start:.2f}s - {current_window_end:.2f}s] (点数 {n_window_points} < {min_samples_needed})")

            # --- 移动到下一个窗口起始时间 ---
            # 防止因步长过小或时间戳重复导致死循环
            if window_step <= 1e-6:
                # 找到严格大于当前起始时间的下一个点的索引
                next_diff_indices = np.where(times > current_window_start)[0]
                if len(next_diff_indices) > 0:
                    current_window_start = times[next_diff_indices[0]]
                else:
                    break # 没有更多不同的时间戳了，结束循环
            else:
                 current_window_start += window_step

             # 特殊处理: 如果这是最后一个可能的窗口起始点，但终点还没覆盖到最后一个数据点，
             # 调整窗口起始点，确保最后一个数据点能被包含在某个窗口中进行处理。
            if current_window_start >= end_time and times[-1] >= current_window_end :
                 # 尝试将窗口向左移动一点，使其包含最后一个点
                 current_window_start = max(start_time, times[-1] - window_duration + 1e-6) # 加一点点防止精度问题

        # --- 滑动窗口处理结束 ---

        num_inliers = np.sum(overall_inlier_mask)
        num_outliers = n_points - num_inliers

        print(f"滑动窗口 RANSAC 完成: 处理了 {processed_windows} 个窗口, 其中 {successful_windows} 个成功拟合。")
        if num_outliers > 0:
            print(f"  滑动窗口 RANSAC: 识别并移除了 {num_outliers} 个离群点 (保留 {num_inliers} / {n_points} 个点).")
        else:
            print("  滑动窗口 RANSAC: 未发现离群点。")

        # 检查过滤后剩余的点数是否足够进行后续处理（例如插值至少需要2个点）
        if num_inliers < 2:
             print(f"警告: 滑动窗口 RANSAC 过滤后剩余的 GPS 点数 ({num_inliers}) 过少 (< 2)，可能导致后续处理失败。将返回过滤后的少量点。")
             # 返回过滤后的点，即使很少
             return times[overall_inlier_mask], positions[overall_inlier_mask]
             # 另一种策略是返回原始数据避免后续崩溃:
             # print(f"警告: ... 将返回原始数据以避免后续处理失败。")
             # return times, positions

        # 返回被识别为内点的时间戳和位置
        return times[overall_inlier_mask], positions[overall_inlier_mask]


def load_gps_data(txt_path: str) -> Dict[str, Any]:
    """加载GPS数据(时间戳, 纬度, 经度, 海拔[, 其他列...]), 进行UTM投影, 并可选地过滤离群点"""
    try:
        # 允许使用空格或逗号作为分隔符
        try:
            gps_data = np.loadtxt(txt_path, delimiter=' ')
        except ValueError:
            gps_data = np.loadtxt(txt_path, delimiter=',')

        if gps_data.ndim == 1:
            gps_data = gps_data.reshape(1, -1)

        assert gps_data.shape[1] >= 4, f"GPS文件需要至少4列 (ts lat lon alt), 找到 {gps_data.shape[1]} 列"

        timestamps = gps_data[:, 0].astype(float)
        lats = gps_data[:, 1].astype(float)
        lons = gps_data[:, 2].astype(float)
        alts = gps_data[:, 3].astype(float)

        # 过滤无效的经纬度值
        valid_gps_mask = (np.abs(lats) <= 90) & (np.abs(lons) <= 180) & (lats != 0) & (lons != 0)
        if not np.any(valid_gps_mask):
             raise ValueError("GPS数据中没有有效的经纬度坐标。")
        initial_count = len(lats)
        if np.sum(valid_gps_mask) < initial_count:
             print(f"警告：过滤了 {initial_count - np.sum(valid_gps_mask)} 个无效的GPS经纬度点。")

        timestamps = timestamps[valid_gps_mask]
        lats = lats[valid_gps_mask]
        lons = lons[valid_gps_mask]
        alts = alts[valid_gps_mask]
        if len(timestamps) == 0: # 如果过滤后没数据了
             raise ValueError("过滤无效经纬度后，GPS数据为空。")


        # === UTM 投影 ===
        utm_zone_number, utm_hemisphere = auto_utm_projection(lons, lats)
        proj_string = f"+proj=utm +zone={utm_zone_number}{utm_hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        try:
             projector = Proj(proj_string)
        except CRSError as e:
             raise ValueError(f"无法创建UTM投影 (Zone: {utm_zone_number}{utm_hemisphere}). 请检查经纬度. Proj Error: {e}")
        x, y = projector(lons, lats)
        utm_positions = np.column_stack((x, y, alts)) # (N, 3) UTM 坐标

        # === 调用 RANSAC 过滤 GPS 离群点 (在UTM坐标上进行) ===
        # *** 注意：现在 filter_gps_outliers_ransac 函数内部会根据配置决定使用全局还是滑动窗口 ***
        print("正在执行 GPS 轨迹 RANSAC 过滤...")
        filtered_times, filtered_utm_positions = filter_gps_outliers_ransac(
            timestamps, utm_positions, CONFIG['gps_filtering_ransac'] # 传递 RANSAC 配置
        )

        # 记录一下过滤掉的点的数量
        ransac_filtered_count = len(timestamps) - len(filtered_times)
        if ransac_filtered_count > 0 :
            print(f"GPS RANSAC 过滤完成，移除了 {ransac_filtered_count} 个点，剩余 {len(filtered_times)} 点。")
        else:
            print(f"GPS RANSAC 过滤完成，未移除点，剩余 {len(filtered_times)} 点。")


        # 如果过滤后数据点过少，无法进行后续处理 (例如插值至少需要2个点)
        if len(filtered_times) < 2:
             raise ValueError("GPS RANSAC 过滤后剩余数据点不足 (< 2)，无法继续处理。请检查 RANSAC 参数设置或原始数据质量。")

        # 使用过滤后的数据进行后续处理
        return {
            'timestamps': filtered_times,        # <-- 使用过滤后的时间戳
            'positions': filtered_utm_positions, # <-- 使用过滤后的UTM位置
            'utm_zone': f"{utm_zone_number}{'S' if utm_hemisphere else 'N'}",
            'projector': projector
        }
    except FileNotFoundError:
        raise ValueError(f"GPS文件未找到: {txt_path}")
    except Exception as e:
        print(f"GPS数据加载、投影或过滤失败 ({txt_path}):")
        traceback.print_exc()
        raise ValueError(f"GPS数据处理失败: {str(e)}")

def utm_to_wgs84(utm_points: np.ndarray, projector: Proj) -> np.ndarray:
    """将UTM坐标批量转换为WGS84经纬度高程"""
    if utm_points.shape[1] != 3:
        raise ValueError("UTM点必须是 Nx3 数组 (X, Y, Z)")
    if not isinstance(projector, Proj):
        raise TypeError("projector 必须是 pyproj.Proj 实例")

    lons, lats = projector(utm_points[:, 0], utm_points[:, 1], inverse=True)
    return np.column_stack((lons, lats, utm_points[:, 2])) # 返回 [lon, lat, alt]

# ----------------------------
# 时间对齐与变换计算
# ----------------------------
def estimate_time_offset(slam_times: np.ndarray, gps_times: np.ndarray, max_samples: int) -> float:
    """通过互相关估计时钟偏移"""
    if len(slam_times) < 2 or len(gps_times) < 2:
        print("警告：SLAM或GPS时间序列过短 (<2)，无法进行可靠的时间偏移估计，返回偏移0。")
        return 0.0

    num_samples = min(max_samples, len(slam_times), len(gps_times))
    slam_sample_times = np.linspace(slam_times.min(), slam_times.max(), num_samples)
    gps_sample_times = np.linspace(gps_times.min(), gps_times.max(), num_samples)

    slam_norm = (slam_sample_times - np.mean(slam_sample_times))
    gps_norm = (gps_sample_times - np.mean(gps_sample_times))
    slam_std = np.std(slam_norm)
    gps_std = np.std(gps_norm)

    if slam_std < 1e-9 or gps_std < 1e-9:
         print("警告：(抽样后)时间戳标准差过小，可能导致互相关不稳定，返回偏移0。")
         return 0.0

    slam_norm /= slam_std
    gps_norm /= gps_std

    corr = np.correlate(slam_norm, gps_norm, mode='full')
    peak_idx = corr.argmax()
    lag = peak_idx - len(slam_norm) + 1

    # 检查 num_samples 是否大于 1
    if num_samples <= 1:
         dt_resampled = 0.0
         print("警告: 用于互相关的样本数不足 (<= 1)，无法计算时间分辨率，偏移可能不准确。")
    else:
         dt_resampled = (slam_sample_times[-1] - slam_sample_times[0]) / (num_samples - 1)

    offset = lag * dt_resampled

    print(f"估计的初始时间偏移: {offset:.3f} 秒")
    return offset

def dynamic_time_alignment(slam_data: Dict[str, np.ndarray],
                           gps_data: Dict[str, np.ndarray],
                           time_align_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    动态时间同步系统 (支持中断处理)：
    1. 估计粗略时间偏移
    2. 检测GPS时间戳中的长时间中断
    3. 对每个连续的GPS数据段分别进行三次样条插值，将其插值到对应的SLAM时间戳上
    返回对齐(插值)后的GPS位置数组 (与SLAM时间戳对应, 中断或无效处为NaN) 和有效性掩码
    """
    slam_times = slam_data['timestamps']
    gps_times = gps_data['timestamps']     # 过滤后的GPS时间
    gps_positions = gps_data['positions'] # 过滤后的UTM位置
    max_corr_samples = time_align_config['max_samples_for_corr']
    max_gps_gap_threshold = time_align_config['max_gps_gap_threshold'] # 获取中断阈值

    n_slam = len(slam_times)
    n_gps = len(gps_times)

    # 初始化输出为无效状态
    aligned_gps_full = np.full((n_slam, 3), np.nan)
    valid_mask = np.zeros(n_slam, dtype=bool)

    if n_slam == 0 or n_gps < 2: # GPS至少需要2个点才能插值
        print(f"警告：SLAM 时间戳为空 ({n_slam}) 或 有效GPS点不足 ({n_gps} < 2)，无法进行时间对齐。")
        return aligned_gps_full, valid_mask

    # 1. 粗粒度偏移估计
    offset = estimate_time_offset(slam_times, gps_times, max_corr_samples)
    adjusted_gps_times = gps_times + offset # 调整GPS时间戳

    # 2. 准备数据：排序、去重
    try:
        sorted_indices = np.argsort(adjusted_gps_times)
        adjusted_gps_times_sorted = adjusted_gps_times[sorted_indices]
        gps_positions_sorted = gps_positions[sorted_indices]

        unique_times, unique_indices = np.unique(adjusted_gps_times_sorted, return_index=True)
        n_unique_gps = len(unique_times)

        if n_unique_gps < 2: # 至少需要2个点才能进行任何插值
             print("警告：去重后的有效GPS时间戳少于2个点，无法进行插值。")
             return aligned_gps_full, valid_mask # 返回全 NaN

        if n_unique_gps < n_gps:
            print(f"警告：移除了 {n_gps - n_unique_gps} 个重复的GPS时间戳。")
            adjusted_gps_times_sorted = unique_times
            gps_positions_sorted = gps_positions_sorted[unique_indices]

        # 3. 检测长时间中断并确定数据段
        time_diffs = np.diff(adjusted_gps_times_sorted)
        gap_indices = np.where(time_diffs > max_gps_gap_threshold)[0] # 找到大于阈值的间隙的 *结束* 索引

        segment_starts = [0] + (gap_indices + 1).tolist() # 每个分段的起始索引 (包括第一个点0)
        segment_ends = gap_indices.tolist() + [n_unique_gps - 1] # 每个分段的结束索引 (包括最后一个点)

        print(f"检测到 {len(gap_indices)} 个 GPS 时间中断 (阈值 > {max_gps_gap_threshold:.1f}s)，将数据分为 {len(segment_starts)} 段进行插值。")

        # 4. 对每个数据段进行插值
        total_valid_points = 0
        for i in range(len(segment_starts)):
            start_idx = segment_starts[i]
            end_idx = segment_ends[i]
            segment_len = end_idx - start_idx + 1

            # 检查段内点数是否足够进行三次样条插值 (至少需要2个点，最好更多)
            # interp1d 'cubic' 严格来说需要 >= 4 个点
            if segment_len < 4:
                kind = 'linear' if segment_len >= 2 else None # 点数不足4个用线性，不足2个无法插值
                if kind is None:
                    # print(f"  跳过段 {i+1}: 点数不足 ({segment_len} < 2)")
                    continue
                # print(f"  警告: 段 {i+1} 点数不足 ({segment_len} < 4)，使用线性插值。")
            else:
                kind = 'cubic'

            # 提取当前段的数据
            segment_times = adjusted_gps_times_sorted[start_idx : end_idx + 1]
            segment_positions = gps_positions_sorted[start_idx : end_idx + 1]
            segment_min_time = segment_times[0]
            segment_max_time = segment_times[-1]

            # 为当前段创建插值函数
            try:
                # **重要**：确保段内时间戳是严格单调递增的，interp1d需要
                if not np.all(np.diff(segment_times) > 0):
                     print(f"警告：段 {i+1} 内时间戳非严格单调递增，跳过此段。")
                     continue

                interp_func_segment = interp1d(
                    segment_times,             # x轴: 当前段的时间
                    segment_positions,         # y轴: 当前段的位置
                    axis=0,                    # 对每一列(x,y,z)独立插值
                    kind=kind,                 # 插值方法: 'cubic' 或 'linear'
                    bounds_error=False,        # 对于超出 *当前段* 时间范围的点不报错
                    fill_value=np.nan          # 对于超出 *当前段* 范围的点填充 NaN
                )
            except ValueError as e:
                print(f"警告：为段 {i+1} 创建插值函数失败 ({kind}插值, {segment_len}点): {e}。跳过此段。")
                continue

            # 找到落入当前段时间范围内的 SLAM 时间戳索引
            # [min, max] 闭区间
            slam_indices_in_segment = np.where(
                (slam_times >= segment_min_time) & (slam_times <= segment_max_time)
            )[0]

            if len(slam_indices_in_segment) > 0:
                # 对这些 SLAM 时间戳进行插值
                interpolated_positions = interp_func_segment(slam_times[slam_indices_in_segment])

                # 将插值结果填充到主数组中
                aligned_gps_full[slam_indices_in_segment] = interpolated_positions
                # 标记这些点为有效 (非NaN)
                # 需要额外检查插值结果是否为NaN (理论上不应发生，除非SL AM时间正好等于段边界且插值出问题)
                non_nan_mask_segment = ~np.isnan(interpolated_positions).any(axis=1)
                valid_mask[slam_indices_in_segment[non_nan_mask_segment]] = True
                total_valid_points += np.sum(non_nan_mask_segment)

        print(f"分段插值完成：在 {n_slam} 个 SLAM 时间点上共生成了 {total_valid_points} 个有效的对齐GPS位置。")
        if total_valid_points == 0:
             print("警告：没有生成任何有效的对齐GPS位置，请检查时间范围重叠、中断阈值或数据质量。")

        return aligned_gps_full, valid_mask

    except ValueError as e:
        print(f"时间对齐或分段插值过程中发生错误: {e}.")
        traceback.print_exc()
        # 返回填充了NaN的数组和全False的掩码
        return np.full((n_slam, 3), np.nan), np.zeros(n_slam, dtype=bool)

# ----------------------------
# 鲁棒变换估计模块
# ----------------------------

def compute_sim3_transform_robust(src: np.ndarray, dst: np.ndarray,
                                 min_samples: int, residual_threshold: float,
                                 max_trials: int, min_inliers_needed: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    使用 RANSAC 稳健地估计源点(src)到目标点(dst)的 Sim3 变换。
    注意：此 RANSAC 在归一化坐标上运行，阈值是相对值。
    失败时返回 None, None, None
    """
    if src.shape[0] < min_samples or dst.shape[0] < min_samples:
        print(f"错误: 点数不足 ({src.shape[0]}) ，无法进行 Sim3 RANSAC (需要至少 {min_samples} 个)")
        return None, None, None
    if src.shape != dst.shape:
         print("错误: Sim3 RANSAC：源点和目标点数量或维度不一致")
         return None, None, None

    # 数据归一化：移至质心，并按平均距离缩放（有助于设定相对阈值）
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    src_norm_factor = np.mean(np.linalg.norm(src_centered, axis=1))
    dst_norm_factor = np.mean(np.linalg.norm(dst_centered, axis=1))

    # 避免除零
    if src_norm_factor < 1e-9 or dst_norm_factor < 1e-9:
        print("警告：Sim3 RANSAC 输入点集分布非常集中，归一化可能不稳定。跳过归一化。")
        src_normalized = src_centered
        dst_normalized = dst_centered
    else:
        src_normalized = src_centered / src_norm_factor
        dst_normalized = dst_centered / dst_norm_factor

    try:
        # 使用 RANSACRegressor 拟合线性变换 (近似 Sim3 中的旋转和缩放部分)
        # 分别对目标坐标的 x, y, z 进行拟合
        ransac_x = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, stop_probability=0.99)
        ransac_y = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, stop_probability=0.99)
        ransac_z = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, stop_probability=0.99)

        ransac_x.fit(src_normalized, dst_normalized[:, 0])
        ransac_y.fit(src_normalized, dst_normalized[:, 1])
        ransac_z.fit(src_normalized, dst_normalized[:, 2])

        # 合并内点掩码：要求在所有维度拟合中都是内点
        inlier_mask = ransac_x.inlier_mask_ & ransac_y.inlier_mask_ & ransac_z.inlier_mask_
        num_inliers = np.sum(inlier_mask)
        print(f"Sim3 RANSAC: 找到 {num_inliers} / {src.shape[0]} 个内点 (阈值={residual_threshold}, 迭代={max_trials})")

        if num_inliers < min_inliers_needed:
            print(f"错误: Sim3 RANSAC: 有效内点不足 ({num_inliers} < {min_inliers_needed})，无法计算可靠的 Sim3 变换")
            return None, None, None

        # 使用 RANSAC 找到的内点来精确计算 Sim3 变换
        return compute_sim3_transform(src[inlier_mask], dst[inlier_mask])

    except ValueError as ve:
         print(f"Sim3 RANSAC 失败: {ve}.")
         return None, None, None
    except Exception as e:
         print(f"Sim3 RANSAC 过程中发生未知错误: {e}.")
         traceback.print_exc()
         return None, None, None

def compute_sim3_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """计算从源点(src)到目标点(dst)的最佳 Sim3 变换 (旋转R, 平移t, 尺度s)。失败时返回 None, None, None"""
    if src.shape[0] < 3:
         print(f"错误: 计算 Sim3 变换需要至少 3 个点，但只有 {src.shape[0]} 个内点")
         return None, None, None
    if src.shape != dst.shape:
         print("错误: Sim3 计算：源点和目标点数量或维度不一致")
         return None, None, None

    try:
        src_centroid = np.mean(src, axis=0)
        dst_centroid = np.mean(dst, axis=0)
        src_centered = src - src_centroid
        dst_centered = dst - dst_centroid

        # 使用 Umeyama 算法的核心思想 (SVD)
        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        V = Vt.T
        R = V @ U.T

        # 处理反射情况 (保证旋转矩阵行列式为 +1)
        if np.linalg.det(R) < 0:
            print("警告: 检测到反射（行列式为负），修正旋转矩阵。")
            Vt_copy = Vt.copy() # 不要修改原始 Vt
            Vt_copy[-1, :] *= -1
            R = Vt_copy.T @ U.T
            # 重新检查行列式
            # print(f"  修正后行列式: {np.linalg.det(R):.3f}")


        # 计算尺度因子 s
        src_dist_sq = np.sum(src_centered**2, axis=1)
        # 计算 dst = s * R @ src 的最佳 s
        # sum(dot(dst_centered[i], s * R @ src_centered[i])) / sum(||s * R @ src_centered[i]||^2)
        # s = sum(dot(dst_centered[i], R @ src_centered[i])) / sum(||src_centered[i]||^2)
        numerator = np.sum(np.sum(dst_centered * (src_centered @ R.T), axis=1)) # sum(dot(dst_i, R @ src_i))
        denominator = np.sum(src_dist_sq) # sum(||src_i||^2)

        if denominator < 1e-9:
             print("警告：源点非常集中于质心，无法可靠计算尺度，默认为 1.0")
             scale = 1.0
        else:
            scale = numerator / denominator
            if scale <= 1e-6:
                 print(f"警告：计算出的尺度非常小 ({scale:.2e})，可能存在问题。重置为 1.0")
                 scale = 1.0

        # 计算平移 t
        t = dst_centroid - scale * (R @ src_centroid)

        print(f"计算得到的 Sim3 参数: scale={scale:.4f}")
        return R, t, scale

    except np.linalg.LinAlgError as e:
        print(f"错误: 计算 Sim3 变换时发生线性代数错误: {e}")
        return None, None, None
    except Exception as e:
        print(f"错误: 计算 Sim3 变换时发生未知错误: {e}")
        traceback.print_exc()
        return None, None, None


# ----------------------------
# 轨迹变换与姿态更新
# ----------------------------

def transform_trajectory(positions: np.ndarray, quaternions: np.ndarray,
                        R: np.ndarray, t: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """应用Sim3变换到整个轨迹（位置和姿态）"""
    # 位置变换: P_new = s * R @ P_old + t
    trans_pos = scale * (positions @ R.T) + t  # 更高效的矩阵运算方式

    # 姿态变换: Q_new = Q_R * Q_old
    # 注意: Sim3 的旋转部分 R 应用于原始姿态
    R_sim3_rot = Rotation.from_matrix(R)
    trans_quat_list = []
    for q_xyzw in quaternions:
        original_rot = Rotation.from_quat(q_xyzw) # scipy quat 是 [x, y, z, w]
        new_rot = R_sim3_rot * original_rot # 旋转合成
        new_quat_xyzw = new_rot.as_quat() # 获取新的 [x, y, z, w]
        trans_quat_list.append(new_quat_xyzw)

    return trans_pos, np.array(trans_quat_list)


# ----------------------------
# 可视化与评估系统
# ----------------------------

def plot_results(original_pos: np.ndarray, corrected_pos: np.ndarray,
                 gps_pos: np.ndarray, # 原始(过滤后)的GPS UTM点
                 valid_indices_for_error: np.ndarray,
                 aligned_gps_for_error: np.ndarray, # 对齐到SLAM时间的GPS点
                 slam_times: np.ndarray, # SLAM 时间戳用于绘图
                 ekf_errors: Optional[np.ndarray] = None # EKF 最终误差 (可能为空)
                ) -> None:
    """增强的可视化系统，显示原始轨迹、修正后轨迹、GPS点和误差"""
    # 设置 matplotlib 支持中文显示
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像时负号'-'显示为方块的问题
    except:
        print("警告：未能设置中文字体 'SimHei'。标签可能显示为方块。请确保已安装该字体。")


    plt.figure(figsize=(15, 12))
    plt.suptitle("SLAM-GPS 轨迹对齐与融合结果", fontsize=16) # 总标题

    # --- 1. 2D轨迹对比 (XY平面) ---
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(original_pos[:, 0], original_pos[:, 1], 'b--', alpha=0.6, linewidth=1, label='原始 SLAM 轨迹')
    ax1.plot(corrected_pos[:, 0], corrected_pos[:, 1], 'g-', linewidth=1.5, label='EKF 修正后轨迹')
    # 绘制原始(过滤后)的GPS点作为参考
    ax1.scatter(gps_pos[:, 0], gps_pos[:, 1], c='r', marker='x', s=30, label='GPS 参考点 (过滤后, UTM)')
    # 绘制用于计算误差的对齐(插值)后的GPS点 (只绘制部分点避免太密集)
    step = max(1, len(valid_indices_for_error) // 100) # 最多绘制约100个对齐点
    if len(valid_indices_for_error) > 0:
       ax1.scatter(aligned_gps_for_error[::step, 0], aligned_gps_for_error[::step, 1],
                   facecolors='none', edgecolors='orange', marker='o', s=40,
                   label='对齐GPS点 (插值, 用于误差评估)')

    ax1.set_title('轨迹对比 (XY平面)')
    ax1.set_xlabel('X (米)')
    ax1.set_ylabel('Y (米)')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal') # 保持XY轴比例一致

    # --- 2. 3D轨迹对比 ---
    ax3d = plt.subplot(2, 2, 2, projection='3d')
    ax3d.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 'b--', alpha=0.6, linewidth=1, label='原始 SLAM 轨迹')
    ax3d.plot(corrected_pos[:, 0], corrected_pos[:, 1], corrected_pos[:, 2], 'g-', linewidth=1.5, label='EKF 修正后轨迹')
    ax3d.scatter(gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2], c='r', marker='x', s=30, label='GPS 参考点 (过滤后, UTM)')
    if len(valid_indices_for_error) > 0:
        ax3d.scatter(aligned_gps_for_error[::step, 0], aligned_gps_for_error[::step, 1], aligned_gps_for_error[::step, 2],
                    facecolors='none', edgecolors='orange', marker='o', s=40,
                    label='对齐GPS点 (插值)')

    ax3d.set_title('轨迹对比 (3D)')
    ax3d.set_xlabel('X (米)')
    ax3d.set_ylabel('Y (米)')
    ax3d.set_zlabel('Z (米)')
    ax3d.legend()
    # 尝试自动调整3D视图范围
    try:
        max_range = np.array([corrected_pos[:,0].max()-corrected_pos[:,0].min(),
                              corrected_pos[:,1].max()-corrected_pos[:,1].min(),
                              corrected_pos[:,2].max()-corrected_pos[:,2].min()]).max() / 2.0 * 1.1 # 稍微扩大一点
        if max_range < 1.0: max_range = 5.0 # 避免范围过小
        mid_x = np.median(corrected_pos[:,0]) # 使用中位数更稳健
        mid_y = np.median(corrected_pos[:,1])
        mid_z = np.median(corrected_pos[:,2])
        ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    except:
        pass # 使用默认视图

    # --- 3. 最终误差分布 (直方图) ---
    ax_hist = plt.subplot(2, 2, 3)
    if ekf_errors is not None and len(ekf_errors) > 0:
        mean_err = np.mean(ekf_errors)
        std_err = np.std(ekf_errors)
        max_err = np.max(ekf_errors)
        median_err = np.median(ekf_errors) # 中位数误差

        ax_hist.hist(ekf_errors, bins=30, alpha=0.75, color='purple')
        ax_hist.axvline(mean_err, color='red', linestyle='dashed', linewidth=1, label=f'均值: {mean_err:.2f}m')
        ax_hist.axvline(median_err, color='orange', linestyle='dashed', linewidth=1, label=f'中位数: {median_err:.2f}m')

        ax_hist.set_title(f'最终位置误差分布 (N={len(ekf_errors)})\n标准差: {std_err:.2f}m, 最大值: {max_err:.2f}m')
        ax_hist.set_xlabel('误差 (米)')
        ax_hist.set_ylabel('频数')
        ax_hist.legend()
        ax_hist.grid(True)
    else:
        ax_hist.set_title("最终位置误差分布")
        ax_hist.text(0.5, 0.5, "无有效误差数据", ha='center', va='center', fontsize=12)
        ax_hist.set_xlabel('误差 (米)')
        ax_hist.set_ylabel('频数')

    # --- 4. 误差随时间变化图 ---
    ax_err_time = plt.subplot(2, 2, 4)
    if ekf_errors is not None and len(valid_indices_for_error) > 0 and len(slam_times) == len(corrected_pos):
        valid_timestamps = slam_times[valid_indices_for_error]
        if len(valid_timestamps) == len(ekf_errors):
             # 计算相对时间 (从第一个有效点开始)
             relative_time = valid_timestamps - valid_timestamps[0]
             ax_err_time.plot(relative_time, ekf_errors, 'r-', linewidth=1, alpha=0.8, label='绝对位置误差')
             ax_err_time.set_xlabel('相对时间 (秒)')
             ax_err_time.set_ylabel('误差 (米)')
             ax_err_time.set_title('误差随时间变化')
             ax_err_time.grid(True)
             ax_err_time.legend()
             ax_err_time.set_ylim(bottom=0) # 误差通常不为负
        else:
             ax_err_time.set_title("误差随时间变化")
             ax_err_time.text(0.5, 0.5, "时间戳与误差数量不匹配", ha='center', va='center', fontsize=10)
             ax_err_time.set_xlabel('相对时间 (秒)')
             ax_err_time.set_ylabel('误差 (米)')

    else:
         ax_err_time.set_title("误差随时间变化")
         ax_err_time.text(0.5, 0.5, "无有效误差数据或时间戳", ha='center', va='center', fontsize=10)
         ax_err_time.set_xlabel('相对时间 (秒)')
         ax_err_time.set_ylabel('误差 (米)')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局，留出总标题空间
    plt.show()

# ----------------------------
# GUI文件选择功能
# ----------------------------

def select_file_dialog(title: str, filetypes: List[Tuple[str, str]]) -> str:
    """通用文件选择对话框"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()   # 关闭Tk实例
    return file_path if file_path else "" # 确保返回字符串

def select_slam_file() -> str:
    """选择SLAM轨迹文件"""
    return select_file_dialog(
        "选择 SLAM 轨迹文件 (TUM格式: timestamp tx ty tz qx qy qz qw)",
        [("文本文件", "*.txt"), ("所有文件", "*.*")]
    )

def select_gps_file() -> str:
    """选择GPS数据文件 (格式: timestamp lat lon alt [...])"""
    return select_file_dialog(
        "选择 GPS 数据文件 (格式: timestamp lat lon alt [...], 空格或逗号分隔)",
        [("文本文件", "*.txt"), ("CSV文件", "*.csv"),("所有文件", "*.*")]
    )

# ----------------------------
# EKF 实现
# ----------------------------

class ExtendedKalmanFilter:
    def __init__(self, initial_pos: np.ndarray, initial_quat: np.ndarray,
                 initial_cov_diag: List[float], process_noise_diag: List[float], meas_noise_diag: List[float]):
        """
        初始化扩展卡尔曼滤波器
        状态向量 (state): [x, y, z, qx, qy, qz, qw] (7维)
        协方差矩阵 (cov): 7x7
        过程噪声 (Q): 7x7, 代表模型预测的不确定性 (例如 SLAM 漂移)
        测量噪声 (R): 3x3, 代表GPS位置测量的不确定性
        """
        if not (initial_pos.shape == (3,) and initial_quat.shape == (4,)):
            raise ValueError(f"EKF 初始化: 初始位置或四元数维度错误 (应为 (3,) 和 (4,), 实际为 {initial_pos.shape} 和 {initial_quat.shape})")

        initial_quat_normalized = self.normalize_quaternion(initial_quat)
        self.state = np.concatenate([initial_pos, initial_quat_normalized]).astype(float)
        self.cov = np.diag(initial_cov_diag).astype(float)
        self.Q = np.diag(process_noise_diag).astype(float)
        self.R = np.diag(meas_noise_diag).astype(float)

        if self.state.shape != (7,): raise ValueError(f"EKF: 内部状态维度错误 (应7, 实际{self.state.shape})")
        if self.cov.shape != (7, 7): raise ValueError(f"EKF: 内部协方差维度错误 (应7x7, 实际{self.cov.shape})")
        if self.Q.shape != (7, 7): raise ValueError(f"EKF: 内部过程噪声Q维度错误 (应7x7, 实际{self.Q.shape})")
        if self.R.shape != (3, 3): raise ValueError(f"EKF: 内部测量噪声R维度错误 (应3x3, 实际{self.R.shape})")

    def predict(self, slam_pos: np.ndarray, slam_quat: np.ndarray, delta_time: float) -> None:
        """预测步骤：直接使用 SLAM 位姿作为预测值，并增加过程噪声"""
        if not (slam_pos.shape == (3,) and slam_quat.shape == (4,)):
             print(f"警告 (EKF Predict): 输入的 SLAM 位姿维度错误 (应为 (3,) 和 (4,), 实际为 {slam_pos.shape} 和 {slam_quat.shape})。跳过此预测步骤。")
             return

        # 状态预测：直接采用 SLAM 的估计值
        self.state[:3] = slam_pos
        self.state[3:] = self.normalize_quaternion(slam_quat)

        # 协方差预测：P = P + Q*dt (简化模型，假设 Q 代表单位时间的噪声增长)
        # 避免 dt 过小或为零导致 Q*dt 失去意义
        dt_adjusted = max(abs(delta_time), 1e-6) # 使用绝对值并设置下限
        self.cov += self.Q * dt_adjusted
        # 确保协方差矩阵保持对称性
        self.cov = (self.cov + self.cov.T) / 2.0

    def update(self, gps_pos: np.ndarray) -> bool:
        """更新步骤：使用有效的 GPS 位置测量值修正状态估计"""
        # 输入检查
        if gps_pos.shape != (3,):
            # print(f"警告 (EKF Update): 输入的 GPS 位置维度错误 (应为 3)，跳过更新。") # 减少冗余打印
            return False
        if np.isnan(gps_pos).any():
             # print(f"警告 (EKF Update): 输入的 GPS 位置包含 NaN，跳过更新。") # 减少冗余打印
             return False

        # 测量矩阵 H (观测模型)：我们只观测位置 x, y, z
        # z = H * state + v
        # [gps_x] = [1 0 0 0 0 0 0] [x ] + [vx]
        # [gps_y] = [0 1 0 0 0 0 0] [y ] + [vy]
        # [gps_z] = [0 0 1 0 0 0 0] [z ] + [vz]
        #                           [qx]
        #                           [qy]
        #                           [qz]
        #                           [qw]
        H = np.zeros((3, 7))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        try:
            # 1. 计算测量残差 (Innovation): y = z_gps - H * x_predicted
            # 由于 H 只选取前三项，等价于 z_gps - x_predicted[:3]
            innovation = gps_pos - self.state[:3]

            # 2. 计算残差协方差 (Innovation Covariance): S = H * P * H^T + R
            # H * P
            H_P = H @ self.cov
            # S = (H * P) * H^T + R
            S = H_P @ H.T + self.R
            S = (S + S.T) / 2.0 # 确保 S 对称

            # 检查 S 是否奇异或接近奇异
            det_S = np.linalg.det(S)
            if abs(det_S) < 1e-12: # 阈值可以调整
                 # print(f"警告 (EKF Update): 残差协方差矩阵 S 接近奇异 (det={det_S:.2e})，可能导致数值不稳定。跳过本次更新。")
                 return False # 跳过更新避免数值问题

            # 3. 计算卡尔曼增益 (Kalman Gain): K = P * H^T * S^(-1)
            # 使用 np.linalg.solve 比直接求逆更稳定: S * K_T = (P * H^T)^T -> S * K_T = H * P^T
            # 或者直接求逆:
            S_inv = np.linalg.inv(S)
            K = self.cov @ H.T @ S_inv

            # 4. 状态更新: x_new = x_predicted + K * y
            self.state += K @ innovation
            # 更新后必须重新归一化四元数部分
            self.state[3:] = self.normalize_quaternion(self.state[3:])

            # 5. 协方差更新: P_new = (I - K * H) * P_predicted
            I = np.eye(7)
            # Joseph form (更稳定): P = (I-KH)P(I-KH)^T + KRK^T
            # P_new = (I - K @ H) @ self.cov @ (I - K @ H).T + K @ self.R @ K.T
            # 标准形式:
            P_new = (I - K @ H) @ self.cov
            # 确保更新后的协方差矩阵仍然是对称半正定的
            self.cov = (P_new + P_new.T) / 2.0

            return True # 更新成功

        except np.linalg.LinAlgError as e:
            print(f"警告 (EKF Update): 更新步骤中发生线性代数错误: {e}。可能是由于数值不稳定。跳过本次更新。")
            # 可以考虑在此处增加协方差矩阵的对角线元素 (P = P + diag(small_value)) 来尝试恢复
            # self.cov += np.eye(7) * 1e-6
            return False
        except Exception as e:
            print(f"警告 (EKF Update): 更新步骤中发生未知错误: {e}。跳过本次更新。")
            traceback.print_exc()
            return False

    @staticmethod
    def normalize_quaternion(q: np.ndarray) -> np.ndarray:
        """归一化四元数，使其模长为1"""
        norm = np.linalg.norm(q)
        if norm < 1e-9: # 避免除以零
            print("警告：尝试归一化零四元数或接近零的四元数，返回默认单位四元数 [0,0,0,1]。")
            return np.array([0.0, 0.0, 0.0, 1.0]) # 返回标准的单位四元数 (w=1)
        return q / norm

# ----------------------------
# EKF 轨迹修正应用函数
# ----------------------------

def apply_ekf_correction(slam_data: Dict[str, np.ndarray], gps_data: Dict[str, np.ndarray],
                         sim3_pos: np.ndarray, sim3_quat: np.ndarray,
                         config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]: # 传递整个 config
    """
    应用 EKF 对经过 Sim3 全局变换的轨迹进行逐点融合修正。
    Args:
        slam_data: 原始 SLAM 数据字典
        gps_data: 原始 (过滤后) GPS 数据字典
        sim3_pos: 经过 Sim3 变换后的 SLAM 位置 (N, 3)
        sim3_quat: 经过 Sim3 变换后的 SLAM 四元数 (N, 4)
        config: 包含 EKF 和 time_alignment 参数的完整配置字典
    Returns:
        Tuple[np.ndarray, np.ndarray]: EKF 修正后的位置 (N, 3) 和四元数 (N, 4)
    """
    n_points = len(slam_data['timestamps'])
    if n_points == 0:
        print("错误 (apply_ekf_correction): 输入的 SLAM 数据为空。")
        return np.empty((0, 3)), np.empty((0, 4))
    if not (sim3_pos.shape[0] == n_points and sim3_quat.shape[0] == n_points):
         raise ValueError(f"错误: Sim3 变换后的轨迹点数 ({sim3_pos.shape[0]}) 与原始 SLAM 时间戳数量 ({n_points}) 不匹配。")


    ekf_config = config['ekf'] # 获取 EKF 配置
    time_align_config = config['time_alignment'] # 获取时间对齐配置

    # 1. 初始化 EKF
    try:
        # 使用 Sim3 变换后的第一个位姿作为初始状态
        ekf = ExtendedKalmanFilter(
            initial_pos=sim3_pos[0],
            initial_quat=sim3_quat[0],
            initial_cov_diag=ekf_config['initial_cov_diag'],
            process_noise_diag=ekf_config['process_noise_diag'],
            meas_noise_diag=ekf_config['meas_noise_diag']
        )
    except Exception as e:
         raise ValueError(f"EKF 初始化失败: {e}")

    # 2. 获取对齐的 GPS 数据用于 EKF 更新 (包含中断处理)
    #    这一步需要在 EKF 循环 *之前* 完成，得到一个与 SLAM 时间戳对应的 GPS 位置数组 (无效处为 NaN)
    print("正在执行分段时间对齐以获取 EKF 更新所需的 GPS 测量...")
    aligned_gps_for_update, valid_mask_for_update = dynamic_time_alignment(
        slam_data, gps_data, time_align_config # 传递时间对齐配置
    )
    # valid_mask_for_update 是一个布尔数组，标记了哪些 SLAM 时间点有有效的对齐 GPS 数据

    num_valid_gps_for_update = np.sum(valid_mask_for_update)
    print(f"EKF 修正：共有 {num_valid_gps_for_update} / {n_points} 个时间点将尝试使用有效的 GPS 测量进行更新。")

    # 准备存储修正后的轨迹
    corrected_pos_list = np.zeros_like(sim3_pos)
    corrected_quat_list = np.zeros_like(sim3_quat)

    # 第一个点的状态就是初始状态
    corrected_pos_list[0] = ekf.state[:3].copy()
    corrected_quat_list[0] = ekf.state[3:].copy()

    last_time = slam_data['timestamps'][0]
    updates_skipped_internal = 0 # 因 EKF 内部问题 (如 S 奇异) 跳过的更新
    updates_applied = 0          # 成功应用的更新次数
    predict_steps = 0            # 执行的预测步骤次数

    # 3. 迭代处理每个 SLAM 时间点 (从第二个点开始)
    for i in range(1, n_points):
        current_time = slam_data['timestamps'][i]
        # 确保 delta_t 不为负或零
        delta_t = current_time - last_time
        if delta_t <= 0:
            print(f"警告 (EKF loop): 时间戳非单调递增或重复在索引 {i} (dt={delta_t:.4f}s)。将使用一个小的正 dt。")
            delta_t = 1e-3 # 使用一个小的正时间间隔

        # a) EKF 预测步骤
        # 使用当前 Sim3 变换后的 SLAM 位姿作为预测输入
        ekf.predict(sim3_pos[i], sim3_quat[i], delta_t)
        predict_steps += 1

        # b) EKF 更新步骤：只在当前时间点有 *有效* (非 NaN) 对齐 GPS 时执行
        if valid_mask_for_update[i]: # 检查该时间点是否有有效的对齐 GPS
            gps_measurement = aligned_gps_for_update[i]
            # EKF 的 update 方法内部会再次检查 NaN (虽然理论上不应发生)
            update_successful = ekf.update(gps_measurement)
            if update_successful:
                 updates_applied += 1
            else:
                 updates_skipped_internal += 1 # EKF 更新失败（例如矩阵奇异）
        # else: # 如果 valid_mask_for_update[i] 是 False (即处于中断或插值无效区域)，则不执行更新

        # 记录当前 EKF 修正后的状态 (无论是否更新，预测后的状态就是当前最佳估计)
        corrected_pos_list[i] = ekf.state[:3].copy()
        corrected_quat_list[i] = ekf.state[3:].copy()
        last_time = current_time

    print(f"EKF 修正完成。共执行 {predict_steps} 步预测。")
    print(f"  在 {num_valid_gps_for_update} 个有有效 GPS 测量的时间点中，成功应用了 {updates_applied} 次更新。")
    if updates_skipped_internal > 0:
        print(f"  因 EKF 内部问题 (如数值不稳定) 跳过了 {updates_skipped_internal} 次更新尝试。")
    no_gps_update_steps = n_points - 1 - num_valid_gps_for_update # -1 因为从1开始循环
    print(f"  共有 {no_gps_update_steps} 步仅执行了 SLAM 预测 (无有效 GPS 可用于更新)。")

    # 再次检查输出长度是否正确 (理论上应该总是匹配)
    if corrected_pos_list.shape[0] != n_points or corrected_quat_list.shape[0] != n_points:
         print(f"严重警告：EKF 输出轨迹长度与输入不匹配！这不应该发生。")
         # 尝试截断或填充，但这可能掩盖根本问题
         corrected_pos_list = corrected_pos_list[:n_points]
         corrected_quat_list = corrected_quat_list[:n_points]

    return corrected_pos_list, corrected_quat_list


# ----------------------------
# 主流程控制
# ----------------------------

def main_process_gui():
    """主处理流程，包含GUI文件选择、处理、评估和可视化"""
    slam_path = ""
    gps_path = ""
    try:
        # 1. 文件选择
        slam_path = select_slam_file()
        if not slam_path: print("未选择 SLAM 文件，操作取消。"); return
        gps_path = select_gps_file()
        if not gps_path: print("未选择 GPS 文件，操作取消。"); return

        print("-" * 30)
        print(f"选择的 SLAM 文件: {slam_path}")
        print(f"选择的 GPS 文件: {gps_path}")
        print("-" * 30)

        # 2. 数据加载与预处理 (包含 GPS RANSAC 过滤)
        print("步骤 1/7: 加载并预处理数据...")
        slam_data = load_slam_trajectory(slam_path)
        # load_gps_data 内部会进行 UTM 投影和 RANSAC 过滤
        gps_data = load_gps_data(gps_path)
        print(f"  SLAM 点数: {len(slam_data['positions'])}")
        print(f"  GPS 点数 (有效、投影并过滤后): {len(gps_data['positions'])}")
        if len(slam_data['positions']) == 0 or len(gps_data['positions']) < 2: # 插值至少需要2个GPS点
             raise ValueError("SLAM 数据为空 或 有效 GPS 数据点不足 (<2)，无法继续处理。")
        print("数据加载与预处理完成。")
        print("-" * 30)

        # 3. 时间对齐 (获取用于 Sim3 的匹配点 - 使用分段对齐)
        print("步骤 2/7: 执行时间对齐以获取 Sim3 匹配点...")
        # 使用 dynamic_time_alignment 获取对齐的 GPS 点和有效性掩码
        aligned_gps_for_sim3, valid_mask_sim3 = dynamic_time_alignment(
            slam_data, gps_data, CONFIG['time_alignment'] # 传递时间对齐配置
        )
        # 获取那些成功对齐的 SLAM 时间点的索引
        valid_indices_sim3 = np.where(valid_mask_sim3)[0]

        # 检查是否有足够的匹配点进行 Sim3 RANSAC
        min_points_needed_for_sim3 = CONFIG['sim3_ransac']['min_inliers_for_transform'] # Sim3 至少需要 4 个内点
        if len(valid_indices_sim3) < min_points_needed_for_sim3:
            raise ValueError(f"有效时间同步匹配点不足 ({len(valid_indices_sim3)} < {min_points_needed_for_sim3})，无法进行 Sim3 变换估计。请检查时间戳、中断阈值或数据质量。")
        print(f"找到 {len(valid_indices_sim3)} 个有效时间同步的匹配点用于 Sim3 估计。")
        print("时间对齐完成。")
        print("-" * 30)


        # 4. Sim3 全局对齐 (使用 RANSAC 找到内点，再用内点计算 Sim3)
        print("步骤 3/7: 计算稳健的 Sim3 全局变换...")
        # 提取用于 Sim3 计算的源点 (SLAM) 和目标点 (对齐后的 GPS)
        src_points = slam_data['positions'][valid_indices_sim3]
        dst_points = aligned_gps_for_sim3[valid_indices_sim3] # 确保只用有效对齐的点

        R, t, scale = compute_sim3_transform_robust(
            src=src_points,
            dst=dst_points,
            min_samples=CONFIG['sim3_ransac']['min_samples'], # RANSAC 内部采样数
            residual_threshold=CONFIG['sim3_ransac']['residual_threshold'], # 归一化空间阈值
            max_trials=CONFIG['sim3_ransac']['max_trials'],
            min_inliers_needed=CONFIG['sim3_ransac']['min_inliers_for_transform'] # 最终接受变换所需内点数
        )

        # 检查 Sim3 计算是否成功
        if R is None or t is None or scale is None:
             # compute_sim3_transform_robust 内部已打印错误信息
             raise RuntimeError("Sim3 全局变换计算失败，无法继续。")

        print("Sim3 全局变换计算成功。")
        print("步骤 4/7: 应用 Sim3 变换到整个 SLAM 轨迹...")
        sim3_pos, sim3_quat = transform_trajectory(
            slam_data['positions'],
            slam_data['quaternions'],
            R, t, scale
        )
        print("Sim3 变换应用完成。")
        print("-" * 30)

        # 5. EKF 局部修正 (融合)
        print("步骤 5/7: 应用 EKF 进行轨迹融合与修正...")
        # 将 Sim3 变换后的轨迹 和 原始(过滤后)的 GPS 数据 送入 EKF
        # apply_ekf_correction 内部会再次调用 dynamic_time_alignment 获取 EKF 更新所需的测量值
        corrected_pos, corrected_quat = apply_ekf_correction(
            slam_data,    # 需要原始时间戳
            gps_data,     # 需要过滤后的 GPS 数据 (时间和位置)
            sim3_pos,     # Sim3 变换后的 SLAM 位置 (作为 EKF 预测输入)
            sim3_quat,    # Sim3 变换后的 SLAM 姿态 (作为 EKF 预测输入)
            CONFIG        # 传递完整的配置字典 (包含 EKF 和时间对齐参数)
        )
        print("EKF 轨迹融合与修正完成。")
        print("-" * 30)

        # 6. 评估误差 (使用最终修正后的轨迹与对齐的GPS点进行比较)
        print("步骤 6/7: 评估轨迹误差...")
        # 需要再次获取对齐的 GPS 点（插值到 SLAM 时间戳上）用于评估
        # 这次获取的对齐点应该与 EKF 更新时内部使用的对齐结果一致
        aligned_gps_for_eval, valid_mask_eval = dynamic_time_alignment(
             slam_data, gps_data, CONFIG['time_alignment'] # 同样需要传递配置
        )
        # 获取那些有有效对齐 GPS 数据的 SLAM 时间戳的索引，用于评估
        valid_indices_eval = np.where(valid_mask_eval)[0]

        # 初始化评估结果
        eval_gps_points = np.empty((0,3)) # 用于绘图的对齐GPS点
        ekf_errors = None                 # EKF 误差数组
        raw_errors = None                 # 原始 SLAM 误差数组
        sim3_errors = None                # Sim3 对齐后误差数组

        if len(valid_indices_eval) > 0:
            # 提取用于评估的对齐后的 GPS 点
            eval_gps_points = aligned_gps_for_eval[valid_indices_eval]

            # --- 使用这些点进行误差比较 ---
            print(f"  (基于 {len(valid_indices_eval)} 个有效对齐点进行评估)")

            # a) 原始 SLAM 轨迹 vs 对齐的 GPS (评估初始误差)
            raw_positions_eval = slam_data['positions'][valid_indices_eval]
            raw_errors = np.linalg.norm(raw_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] 原始轨迹   vs 对齐GPS -> "
                  f"均值误差: {np.mean(raw_errors):.3f} m, "
                  f"中位数误差: {np.median(raw_errors):.3f} m, "
                  f"标准差: {np.std(raw_errors):.3f} m, "
                  f"最大误差: {np.max(raw_errors):.3f} m")

            # b) Sim3 对齐后 vs 对齐的 GPS (评估全局对齐效果)
            sim3_positions_eval = sim3_pos[valid_indices_eval]
            sim3_errors = np.linalg.norm(sim3_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] Sim3 对齐后 vs 对齐GPS -> "
                  f"均值误差: {np.mean(sim3_errors):.3f} m, "
                  f"中位数误差: {np.median(sim3_errors):.3f} m, "
                  f"标准差: {np.std(sim3_errors):.3f} m, "
                  f"最大误差: {np.max(sim3_errors):.3f} m")

            # c) EKF 融合后 vs 对齐的 GPS (评估最终效果)
            ekf_positions_eval = corrected_pos[valid_indices_eval]
            ekf_errors = np.linalg.norm(ekf_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] EKF 融合后  vs 对齐GPS -> "
                  f"均值误差: {np.mean(ekf_errors):.3f} m, "
                  f"中位数误差: {np.median(ekf_errors):.3f} m, "
                  f"标准差: {np.std(ekf_errors):.3f} m, "
                  f"最大误差: {np.max(ekf_errors):.3f} m")
        else:
            print("警告：无有效对齐的GPS点用于最终误差评估 (可能全程中断或时间不重叠)。")
            # 确保 ekf_errors 为 None 或空数组，以便绘图函数处理
            ekf_errors = None # 或者 np.array([])

        print("误差评估完成。")
        print("-" * 30)


        # 7. 保存结果
        print("步骤 7/7: 保存结果与可视化...")
        save_results = messagebox.askyesno("保存结果", "处理完成。是否要保存修正后的轨迹?")
        if save_results:
            # 默认文件名建议
            base_filename = slam_path.split('/')[-1].split('\\')[-1] # 获取原始文件名
            default_save_name = base_filename.replace('.txt', '_corrected_utm.txt')
            if default_save_name == base_filename: # 如果没有 .txt 后缀
                 default_save_name = base_filename + '_corrected_utm.txt'

            output_path_utm = filedialog.asksaveasfilename(
                title="保存修正后的轨迹 (UTM 坐标, TUM 格式)",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt")],
                initialfile=default_save_name
            )
            if output_path_utm:
                # 准备 TUM 格式数据: timestamp tx ty tz qx qy qz qw
                output_data_utm = np.column_stack((
                    slam_data['timestamps'],
                    corrected_pos,       # EKF 修正后的 UTM 位置
                    corrected_quat       # EKF 修正后的四元数
                ))
                np.savetxt(output_path_utm, output_data_utm,
                           fmt=['%.6f'] + ['%.6f'] * 3 + ['%.8f'] * 4, # 位置6位小数，四元数8位
                           header="timestamp x y z qx qy qz qw (UTM)",
                           comments='')
                print(f"  UTM 坐标轨迹已保存至: {output_path_utm}")

                # 尝试保存 WGS84 格式
                try:
                    projector = gps_data.get('projector')
                    if projector:
                        print("  正在将修正后的轨迹转换回 WGS84 坐标...")
                        wgs84_lon_lat_alt = utm_to_wgs84(corrected_pos, projector)
                        # 准备 WGS84 格式数据: timestamp lon lat alt qx qy qz qw
                        output_data_wgs84 = np.column_stack((
                            slam_data['timestamps'],
                            wgs84_lon_lat_alt[:, 0], # lon
                            wgs84_lon_lat_alt[:, 1], # lat
                            wgs84_lon_lat_alt[:, 2], # alt
                            corrected_quat           # EKF 修正后的四元数
                        ))
                        # 生成 WGS84 文件名
                        output_path_wgs84 = output_path_utm.replace('_utm.txt', '_wgs84.txt')
                        if output_path_wgs84 == output_path_utm: # 以防万一没替换成功
                             output_path_wgs84 = output_path_utm.replace('.txt', '_wgs84.txt')

                        np.savetxt(output_path_wgs84, output_data_wgs84,
                                   fmt=['%.6f'] + ['%.8f', '%.8f', '%.3f'] + ['%.8f'] * 4, # lon/lat 8位小数, alt 3位, quat 8位
                                   header="timestamp lon lat alt qx qy qz qw (WGS84)",
                                   comments='')
                        print(f"  WGS84 坐标轨迹已保存至: {output_path_wgs84}")
                    else:
                         print("警告：GPS 数据中缺少投影仪对象 'projector'，无法自动保存 WGS84 格式轨迹。")
                except Exception as e:
                    print(f"错误：保存 WGS84 格式轨迹失败: {str(e)}")
                    traceback.print_exc()
            else:
                print("  用户取消保存。")


        # 8. 可视化 (传递必要的评估结果给绘图函数)
        print("  正在生成可视化结果图...")
        plot_results(
            original_pos=slam_data['positions'], # 原始 SLAM 位置
            corrected_pos=corrected_pos,         # EKF 修正后的位置
            gps_pos=gps_data['positions'],       # 过滤后的原始 GPS 点 (UTM)
            valid_indices_for_error=valid_indices_eval, # 用于评估的有效点索引
            aligned_gps_for_error=eval_gps_points,      # 用于评估的对齐 GPS 点
            slam_times=slam_data['timestamps'],         # SLAM 时间戳
            ekf_errors=ekf_errors                       # EKF 最终误差数组 (可能为 None)
        )

        print("-" * 30)
        print("所有处理步骤完成。")
        messagebox.showinfo("完成", "轨迹对齐与融合处理完成！")

    except ValueError as ve:
        error_msg = f"处理失败 (值错误): {str(ve)}"
        print(f"\n错误!\n{error_msg}")
        if gps_path and not slam_path: error_msg += f"\nGPS 文件: {gps_path}"
        if slam_path and not gps_path: error_msg += f"\nSLAM 文件: {slam_path}"
        if slam_path and gps_path: error_msg += f"\nSLAM 文件: {slam_path}\nGPS 文件: {gps_path}"
        traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except FileNotFoundError as fnf:
        error_msg = f"处理失败 (文件未找到): {str(fnf)}"
        print(f"\n错误!\n{error_msg}"); messagebox.showerror("处理失败", error_msg)
    except AssertionError as ae:
        error_msg = f"处理失败 (数据格式断言错误): {str(ae)}"
        print(f"\n错误!\n{error_msg}")
        if gps_path and not slam_path: error_msg += f"\nGPS 文件: {gps_path}"
        if slam_path and not gps_path: error_msg += f"\nSLAM 文件: {slam_path}"
        if slam_path and gps_path: error_msg += f"\nSLAM 文件: {slam_path}\nGPS 文件: {gps_path}"
        traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except CRSError as ce:
         error_msg = f"处理失败 (坐标投影错误): {str(ce)}"
         print(f"\n错误!\n{error_msg}")
         traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except np.linalg.LinAlgError as lae:
         error_msg = f"处理失败 (线性代数计算错误): {str(lae)}"
         print(f"\n错误!\n{error_msg}")
         traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except RuntimeError as rte: # 捕获由 Sim3 失败等引发的 RuntimeError
         error_msg = f"处理失败 (运行时错误): {str(rte)}"
         print(f"\n错误!\n{error_msg}")
         messagebox.showerror("处理失败", error_msg)
    except Exception as e:
        error_msg = f"处理过程中发生未预料的错误: {type(e).__name__}: {str(e)}"
        print(f"\n严重错误!\n{error_msg}")
        if gps_path and not slam_path: error_msg += f"\nGPS 文件: {gps_path}"
        if slam_path and not gps_path: error_msg += f"\nSLAM 文件: {slam_path}"
        if slam_path and gps_path: error_msg += f"\nSLAM 文件: {slam_path}\nGPS 文件: {gps_path}"
        traceback.print_exc()
        messagebox.showerror("处理失败", f"{error_msg}\n\n详情请查看控制台输出。")

if __name__ == "__main__":
    print("启动 SLAM-GPS 轨迹对齐与融合工具 (支持滑动窗口RANSAC与GPS中断处理)...")
    print("="*70)
    print("配置参数概览:")
    print(f"  GPS RANSAC 滤波启用: {CONFIG['gps_filtering_ransac']['enabled']}")
    if CONFIG['gps_filtering_ransac']['enabled']:
        use_sw = CONFIG['gps_filtering_ransac']['use_sliding_window']
        print(f"  GPS RANSAC 模式: {'滑动窗口' if use_sw else '全局'}")
        if use_sw:
            print(f"    窗口时长: {CONFIG['gps_filtering_ransac']['window_duration_seconds']} s")
            print(f"    窗口步长因子: {CONFIG['gps_filtering_ransac']['window_step_factor']}")
        print(f"    多项式阶数: {CONFIG['gps_filtering_ransac']['polynomial_degree']}")
        print(f"    最小样本数: {CONFIG['gps_filtering_ransac']['min_samples']}")
        print(f"    残差阈值: {CONFIG['gps_filtering_ransac']['residual_threshold_meters']} m")
        print(f"    最大迭代次数: {CONFIG['gps_filtering_ransac']['max_trials']}")
    print(f"  GPS 中断阈值: {CONFIG['time_alignment']['max_gps_gap_threshold']} s")
    print(f"  Sim3 RANSAC 最小内点数: {CONFIG['sim3_ransac']['min_inliers_for_transform']}")
    print("="*70)

    main_process_gui()
    print("\n程序结束。")

