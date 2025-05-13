# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from pyproj import Proj
from pyproj.exceptions import CRSError
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, messagebox
import traceback
from typing import Dict, Tuple, Optional, List, Any

# ----------------------------
# 配置参数 
# ----------------------------
CONFIG = {
    # EKF 参数
    "ekf": {
        "initial_cov_diag": [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01], # x,y,z, qx,qy,qz,qw
        "process_noise_diag": [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01], # Process noise per second (approx) for x,y,z, qx,qy,qz,qw
        "meas_noise_diag": [0.2, 0.2, 0.2], # GPS x,y,z measurement noise std dev
        "transition_steps": 20,         # GNSS 恢复时，平滑过渡所需的步数
    },
    # Sim(3) 全局变换 RANSAC 参数
    "sim3_ransac": {
        "min_samples": 4,               # RANSAC 模型最小样本数
        "residual_threshold": 4.0,      # RANSAC 内点阈值 (米)
        "max_trials": 1000,             # RANSAC 最大迭代次数
        "min_inliers_needed": 4,        # 有效 Sim3 拟合所需的最小内点数
        "max_initial_duration": 120.0,   # <<< 新增: 用于计算 Sim3 的第一段匹配的最大时长 (秒)
    },
    # GPS 轨迹 RANSAC 滤波参数
    "gps_filtering_ransac": {
        "enabled": True,
        "use_sliding_window": True,
        "window_duration_seconds": 15.0,
        "window_step_factor": 0.5,
        "polynomial_degree": 2,
        "min_samples": 6,
        "residual_threshold_meters": 10.0,
        "max_trials": 50,
    },
    # 时间对齐参数
    "time_alignment": {
        "max_samples_for_corr": 500,    # 初始时间偏移互相关最大样本数
        "max_gps_gap_threshold": 5.0,   # 定义 GPS 段中断的最大时间间隙 (秒)
    }
}

# ----------------------------
# 辅助函数
# ----------------------------

def calculate_relative_pose(pose1_pos: np.ndarray, pose1_quat: np.ndarray,
                            pose2_pos: np.ndarray, pose2_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算从 pose1 到 pose2 的相对运动。

    Args:
        pose1_pos (np.ndarray): 起始位置 [x, y, z]
        pose1_quat (np.ndarray): 起始姿态四元数 [x, y, z, w]
        pose2_pos (np.ndarray): 结束位置 [x, y, z]
        pose2_quat (np.ndarray): 结束姿态四元数 [x, y, z, w]

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - delta_pos_local (np.ndarray): 在 pose1 坐标系下的位置变化 [dx, dy, dz]
            - delta_quat (np.ndarray): 相对旋转四元数 [x, y, z, w] (从 pose1 到 pose2)
    """
    try:
        rot1 = Rotation.from_quat(pose1_quat)
        rot1_inv = rot1.inv()
        rot2 = Rotation.from_quat(pose2_quat)
    except ValueError as e:
        print(f"警告 (calculate_relative_pose): 无效的四元数输入: {e}. 返回零运动。")
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]) # Zero translation, identity rotation

    delta_pos_world = pose2_pos - pose1_pos
    delta_pos_local = rot1_inv.apply(delta_pos_world)
    delta_rot = rot1_inv * rot2
    delta_quat = delta_rot.as_quat() # [x, y, z, w]

    return delta_pos_local, delta_quat

def quaternion_nlerp(q1: np.ndarray, q2: np.ndarray, weight_q2: float) -> np.ndarray:
    """
    对两个四元数进行归一化线性插值 (NLERP)。
    weight_q2 是 q2 的权重 (范围 0 到 1)。
    """
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2 # 反转 q2 以保证插值走最短路径
        dot = -dot

    w = np.clip(weight_q2, 0.0, 1.0)
    q_interp = (1.0 - w) * q1 + w * q2

    norm = np.linalg.norm(q_interp)
    if norm < 1e-9:
        # 处理插值结果接近零的情况
        return q1 if weight_q2 < 0.5 else q2
    return q_interp / norm

# ----------------------------
# 数据加载与预处理
# ----------------------------

def load_slam_trajectory(txt_path: str) -> Dict[str, np.ndarray]:
    """加载并验证SLAM轨迹数据 (TUM格式)"""
    try:
        data = np.loadtxt(txt_path)
        if data.ndim == 1:
             data = data.reshape(1, -1)
        if data.shape[1] != 8:
             raise ValueError(f"SLAM文件格式错误：需要8列 (ts x y z qx qy qz qw), 找到 {data.shape[1]} 列")
        return {
            'timestamps': data[:, 0].astype(float),
            'positions': data[:, 1:4].astype(float),
            'quaternions': data[:, 4:8].astype(float) # Scipy 使用 [x, y, z, w]
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
# GPS 离群点过滤模块
# -------------------------------------------
def filter_gps_outliers_ransac(times: np.ndarray, positions: np.ndarray,
                               config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 RANSAC 和多项式模型过滤 GPS 轨迹中的离群点。
    支持全局拟合或滑动窗口局部拟合。
    """
    if not config.get("enabled", False):
        print("GPS RANSAC 过滤被禁用。")
        return times, positions

    n_points = len(times)
    min_samples_needed = config['min_samples']

    if n_points < min_samples_needed:
        print(f"警告: GPS 点数 ({n_points}) 少于 RANSAC 最小样本数 ({min_samples_needed})，跳过 GPS 离群点过滤。")
        return times, positions

    use_sliding_window = config.get("use_sliding_window", False)

    if not use_sliding_window:
        # --- 全局 RANSAC 拟合 ---
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
        # --- 滑动窗口 RANSAC 拟合 ---
        window_duration = config['window_duration_seconds']
        step_factor = config['window_step_factor']
        window_step = window_duration * step_factor
        residual_threshold = config['residual_threshold_meters']
        poly_degree = config['polynomial_degree']
        max_trials_per_window = config['max_trials']

        print(f"正在执行滑动窗口 GPS RANSAC 过滤 (窗口时长: {window_duration}s, 步长: {window_step:.2f}s)...")

        if n_points < min_samples_needed:
             print(f"警告：总点数 ({n_points}) 少于窗口最小样本数 ({min_samples_needed})，无法使用滑动窗口。")
             return times, positions

        overall_inlier_mask = np.zeros(n_points, dtype=bool)
        processed_windows = 0
        successful_windows = 0

        start_time = times[0]
        end_time = times[-1]
        current_window_start = start_time

        while current_window_start < end_time:
            current_window_end = current_window_start + window_duration
            # 找出时间戳落在当前窗口内的点的索引
            window_indices = np.where((times >= current_window_start) & (times < current_window_end))[0]
            n_window_points = len(window_indices)

            # 只有当窗口内点数足够时才进行 RANSAC
            if n_window_points >= min_samples_needed:
                processed_windows += 1
                window_times = times[window_indices]
                window_positions = positions[window_indices]
                window_t_feature = window_times.reshape(-1, 1)

                try:
                    window_inlier_masks_dim = []
                    valid_window_fit = True
                    for i in range(positions.shape[1]): # 对 X, Y, Z 分别拟合
                        target = window_positions[:, i]
                        model = make_pipeline(
                            PolynomialFeatures(degree=poly_degree),
                            RANSACRegressor(
                                min_samples=min_samples_needed,
                                residual_threshold=residual_threshold,
                                max_trials=max_trials_per_window,
                            )
                        )
                        model.fit(window_t_feature, target)
                        inlier_mask_window_dim = model[-1].inlier_mask_
                        window_inlier_masks_dim.append(inlier_mask_window_dim)

                    # 只有 X,Y,Z 都成功拟合才认为是有效窗口
                    if valid_window_fit:
                        # 要求 X,Y,Z 都是内点
                        window_final_inlier_mask = np.logical_and.reduce(window_inlier_masks_dim)
                        # 获取这些内点在原始数据中的索引
                        original_indices_of_inliers = window_indices[window_final_inlier_mask]
                        # 在全局掩码中标记这些内点
                        overall_inlier_mask[original_indices_of_inliers] = True
                        successful_windows += 1

                except Exception as e:
                     # 如果某个窗口拟合失败，打印警告并继续下一个窗口
                     print(f"警告：窗口 [{current_window_start:.2f}s - {current_window_end:.2f}s] RANSAC 拟合失败: {e}")

            # 移动到下一个窗口起始时间
            if window_step <= 1e-6: # 防止步长过小导致死循环
                # 如果步长为0或负数，则移动到下一个不同的时间戳
                next_diff_indices = np.where(times > current_window_start)[0]
                if len(next_diff_indices) > 0:
                    current_window_start = times[next_diff_indices[0]]
                else:
                    break # 没有更多时间戳了
            else:
                 current_window_start += window_step

            # 确保最后一个窗口能包含最后一个数据点
            if current_window_start >= end_time and times[-1] >= current_window_end :
                 # 如果已经处理完最后一个点，将窗口起始时间调整到能包含最后一个点的位置
                 current_window_start = max(start_time, times[-1] - window_duration + 1e-6)


        num_inliers = np.sum(overall_inlier_mask)
        num_outliers = n_points - num_inliers

        print(f"滑动窗口 RANSAC 完成: 处理了 {processed_windows} 个窗口, 其中 {successful_windows} 个成功拟合。")
        if num_outliers > 0:
            print(f"  滑动窗口 RANSAC: 识别并移除了 {num_outliers} 个离群点 (保留 {num_inliers} / {n_points} 个点).")
        else:
            print("  滑动窗口 RANSAC: 未发现离群点。")

        if num_inliers < 2: # 后续插值至少需要2个点
             print(f"警告: 滑动窗口 RANSAC 过滤后剩余的 GPS 点数 ({num_inliers}) 过少 (< 2)，可能导致后续处理失败。将返回过滤后的少量点。")
             # 即使点少也返回，让后续步骤决定如何处理
             return times[overall_inlier_mask], positions[overall_inlier_mask]

        return times[overall_inlier_mask], positions[overall_inlier_mask]


def load_gps_data(txt_path: str) -> Dict[str, Any]:
    """加载GPS数据(时间戳, 纬度, 经度, 海拔[, 其他列...]), 进行UTM投影, 并可选地过滤离群点"""
    try:
        # 尝试用空格或逗号作为分隔符加载
        try:
            gps_data = np.loadtxt(txt_path, delimiter=' ')
        except ValueError:
            gps_data = np.loadtxt(txt_path, delimiter=',')

        if gps_data.ndim == 1:
            gps_data = gps_data.reshape(1, -1)

        if gps_data.shape[1] < 4:
             raise ValueError(f"GPS文件需要至少4列 (ts lat lon alt), 找到 {gps_data.shape[1]} 列")

        timestamps = gps_data[:, 0].astype(float)
        lats = gps_data[:, 1].astype(float)
        lons = gps_data[:, 2].astype(float)
        alts = gps_data[:, 3].astype(float)

        # 过滤无效的经纬度值 (纬度[-90, 90], 经度[-180, 180], 且不为0)
        valid_gps_mask = (np.abs(lats) <= 90) & (np.abs(lons) <= 180) & (lats != 0) & (lons != 0)
        initial_count = len(lats)
        if not np.all(valid_gps_mask):
            num_filtered = initial_count - np.sum(valid_gps_mask)
            print(f"警告：过滤了 {num_filtered} 个无效的GPS经纬度点（纬度或经度超出范围或为0）。")
            timestamps = timestamps[valid_gps_mask]
            lats = lats[valid_gps_mask]
            lons = lons[valid_gps_mask]
            alts = alts[valid_gps_mask]
            if len(timestamps) == 0:
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
        print("正在执行 GPS 轨迹 RANSAC 过滤...")
        filtered_times, filtered_utm_positions = filter_gps_outliers_ransac(
            timestamps, utm_positions, CONFIG['gps_filtering_ransac'] # 传递 RANSAC 配置
        )

        ransac_filtered_count = len(timestamps) - len(filtered_times)
        if ransac_filtered_count > 0 :
            print(f"GPS RANSAC 过滤完成，移除了 {ransac_filtered_count} 个点，剩余 {len(filtered_times)} 点。")
        else:
            print(f"GPS RANSAC 过滤完成，未移除点，剩余 {len(filtered_times)} 点。")

        if len(filtered_times) < 2: # 插值至少需要2个点
             raise ValueError("GPS RANSAC 过滤后剩余数据点不足 (< 2)，无法继续处理。请检查 RANSAC 参数设置或原始数据质量。")

        return {
            'timestamps': filtered_times,        # 使用过滤后的时间戳
            'positions': filtered_utm_positions, # 使用过滤后的UTM位置
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

    # 限制用于互相关的样本数量，避免过大计算量
    num_samples = min(max_samples, len(slam_times), len(gps_times))
    # 在各自时间范围内均匀采样
    slam_sample_times = np.linspace(slam_times.min(), slam_times.max(), num_samples)
    gps_sample_times = np.linspace(gps_times.min(), gps_times.max(), num_samples)

    # 归一化时间戳序列以进行互相关
    slam_norm = (slam_sample_times - np.mean(slam_sample_times))
    gps_norm = (gps_sample_times - np.mean(gps_sample_times))
    slam_std = np.std(slam_norm)
    gps_std = np.std(gps_norm)

    # 避免除以零
    if slam_std < 1e-9 or gps_std < 1e-9:
         print("警告：(抽样后)时间戳标准差过小，可能导致互相关不稳定，返回偏移0。")
         return 0.0

    slam_norm /= slam_std
    gps_norm /= gps_std

    # 计算互相关
    corr = np.correlate(slam_norm, gps_norm, mode='full')
    # 找到相关峰值对应的延迟（lag）
    peak_idx = corr.argmax()
    lag = peak_idx - len(slam_norm) + 1 # 互相关结果的中心对应 lag=0

    # 计算重采样后的时间分辨率
    if num_samples <= 1:
         dt_resampled = 0.0 # 无法计算分辨率
         print("警告: 用于互相关的样本数不足 (<= 1)，无法计算时间分辨率，偏移可能不准确。")
    else:
         dt_resampled = (slam_sample_times[-1] - slam_sample_times[0]) / (num_samples - 1)

    # 估计的时间偏移 = 延迟 * 时间分辨率
    offset = lag * dt_resampled
    print(f"估计的初始时间偏移: {offset:.3f} 秒 (正值表示 GPS 时间戳晚于 SLAM)")
    return offset

def dynamic_time_alignment(slam_data: Dict[str, np.ndarray],
                           gps_data: Dict[str, np.ndarray],
                           time_align_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    动态时间同步系统 (支持中断处理)：
    对每个连续的GPS数据段分别进行插值，将其插值到对应的SLAM时间戳上。

    Args:
        slam_data: 包含 'timestamps', 'positions', 'quaternions' 的字典
        gps_data: 包含 'timestamps', 'positions' (过滤后UTM) 的字典
        time_align_config: 包含 'max_samples_for_corr', 'max_gps_gap_threshold' 的字典

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - aligned_gps_full (np.ndarray): (N_slam, 3) 数组，在每个 SLAM 时间戳上插值得到的 GPS 位置，无效处为 NaN
            - valid_mask (np.ndarray): (N_slam,)布尔数组，标记哪些 SLAM 时间戳有有效的插值结果
    """
    slam_times = slam_data['timestamps']
    gps_times = gps_data['timestamps']     # 过滤后的GPS时间
    gps_positions = gps_data['positions'] # 过滤后的UTM位置
    max_corr_samples = time_align_config['max_samples_for_corr']
    max_gps_gap_threshold = time_align_config['max_gps_gap_threshold'] # 获取中断阈值

    n_slam = len(slam_times)
    n_gps = len(gps_times)

    # 初始化结果数组，默认所有位置无效
    aligned_gps_full = np.full((n_slam, 3), np.nan)
    valid_mask = np.zeros(n_slam, dtype=bool)

    # 如果 SLAM 数据为空或 GPS 数据点过少，无法进行对齐
    if n_slam == 0 or n_gps < 2: # 插值至少需要2个点
        print(f"警告：SLAM 时间戳为空 ({n_slam}) 或 有效GPS点不足 ({n_gps} < 2)，无法进行时间对齐。")
        return aligned_gps_full, valid_mask

    # 1. 估计初始时间偏移
    offset = estimate_time_offset(slam_times, gps_times, max_corr_samples)
    adjusted_gps_times = gps_times + offset # 调整GPS时间戳

    try:
        # 2. 对调整后的 GPS 时间戳进行排序，并处理重复时间戳
        sorted_indices = np.argsort(adjusted_gps_times)
        adjusted_gps_times_sorted = adjusted_gps_times[sorted_indices]
        gps_positions_sorted = gps_positions[sorted_indices]

        # 使用 unique 移除重复时间戳，保留第一个出现的索引
        unique_times, unique_indices = np.unique(adjusted_gps_times_sorted, return_index=True)
        n_unique_gps = len(unique_times)

        if n_unique_gps < 2:
             print("警告：去重后的有效GPS时间戳少于2个点，无法进行插值。")
             return aligned_gps_full, valid_mask

        if n_unique_gps < n_gps:
            print(f"警告：移除了 {n_gps - n_unique_gps} 个重复的GPS时间戳。")
            adjusted_gps_times_sorted = unique_times
            gps_positions_sorted = gps_positions_sorted[unique_indices]

        # 3. 检测 GPS 时间中断，分割数据段
        time_diffs = np.diff(adjusted_gps_times_sorted)
        # 找到时间差大于阈值的索引，这些索引是中断 *之后* 的第一个点的索引
        gap_indices = np.where(time_diffs > max_gps_gap_threshold)[0]

        # 定义每个连续段的起始和结束索引
        segment_starts = [0] + (gap_indices + 1).tolist() # 每个分段的起始索引 (包括第一个点0)
        segment_ends = gap_indices.tolist() + [n_unique_gps - 1] # 每个分段的结束索引 (包括最后一个点)

        num_segments = len(segment_starts)
        print(f"检测到 {num_segments - 1} 个 GPS 时间中断 (阈值 > {max_gps_gap_threshold:.1f}s)，将数据分为 {num_segments} 段进行插值。")

        # 4. 对每个分段进行插值
        total_valid_points = 0
        for i in range(num_segments):
            start_idx = segment_starts[i]
            end_idx = segment_ends[i]
            segment_len = end_idx - start_idx + 1

            # 选择插值方法：点数少于4用线性，否则用三次样条
            if segment_len < 4:
                kind = 'linear' if segment_len >= 2 else None # 点数不足2个无法插值
                if kind is None:
                    # print(f"  段 {i+1} 点数 ({segment_len}) 不足2，跳过。")
                    continue
            else:
                kind = 'cubic'

            segment_times = adjusted_gps_times_sorted[start_idx : end_idx + 1]
            segment_positions = gps_positions_sorted[start_idx : end_idx + 1]
            segment_min_time = segment_times[0]
            segment_max_time = segment_times[-1]

            # 确保段内时间戳严格单调递增 (interp1d 要求)
            if not np.all(np.diff(segment_times) > 0):
                 print(f"警告：段 {i+1} 内时间戳非严格单调递增，跳过此段。")
                 continue

            # 创建插值函数
            try:
                interp_func_segment = interp1d(
                    segment_times, segment_positions, axis=0, kind=kind,
                    bounds_error=False, # 不对超出范围的时间戳报错
                    fill_value=np.nan   # 超出范围的时间戳返回 NaN
                )
            except ValueError as e:
                print(f"警告：为段 {i+1} 创建插值函数失败 ({kind}插值, {segment_len}点): {e}。跳过此段。")
                continue

            # 找出落在当前 GPS 段时间范围内的 SLAM 时间戳
            slam_indices_in_segment = np.where(
                (slam_times >= segment_min_time) & (slam_times <= segment_max_time)
            )[0]

            # 对这些 SLAM 时间戳进行插值
            if len(slam_indices_in_segment) > 0:
                interpolated_positions = interp_func_segment(slam_times[slam_indices_in_segment])

                # 将插值结果填入全局结果数组
                aligned_gps_full[slam_indices_in_segment] = interpolated_positions

                # 更新有效掩码 (标记非 NaN 的结果)
                non_nan_mask_segment = ~np.isnan(interpolated_positions).any(axis=1)
                valid_indices_in_segment = slam_indices_in_segment[non_nan_mask_segment]
                valid_mask[valid_indices_in_segment] = True
                total_valid_points += len(valid_indices_in_segment)

        print(f"分段插值完成：在 {n_slam} 个 SLAM 时间点上共生成了 {total_valid_points} 个有效的对齐GPS位置。")
        if total_valid_points == 0:
             print("警告：没有生成任何有效的对齐GPS位置，请检查时间范围重叠、中断阈值或数据质量。")

        return aligned_gps_full, valid_mask

    except ValueError as e:
        print(f"时间对齐或分段插值过程中发生错误: {e}.")
        traceback.print_exc()
        # 返回空的/无效的结果
        return np.full((n_slam, 3), np.nan), np.zeros(n_slam, dtype=bool)


def compute_sim3_transform_robust(src: np.ndarray, dst: np.ndarray,
                                  min_samples: int, residual_threshold: float,
                                  max_trials: int, min_inliers_needed: int,
                                  point_description: str = "points" # 新增描述参数
                                  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    使用 RANSAC 稳健地估计源点(src)到目标点(dst)的 Sim3 变换。
    Args:
        src: 源点坐标 (N, 3)
        dst: 目标点坐标 (N, 3)
        min_samples: RANSAC 最小样本数
        residual_threshold: RANSAC 内点残差阈值
        max_trials: RANSAC 最大迭代次数
        min_inliers_needed: 接受结果所需的最小内点数
        point_description: 用于打印信息的点描述 (例如 "initial segment points")
    Returns:
        (R, t, scale) or (None, None, None) if failed
    """
    n_points = src.shape[0]
    if n_points < min_samples:
        print(f"错误: Sim3 RANSAC: 输入点数 ({n_points} from {point_description}) 不足，无法进行 RANSAC (需要至少 {min_samples} 个)")
        return None, None, None
    if src.shape != dst.shape:
         print(f"错误: Sim3 RANSAC: 源点和目标点 ({point_description}) 数量或维度不一致 ({src.shape} vs {dst.shape})")
         return None, None, None

    # --- RANSAC 过程 ---
    best_inlier_mask = None
    max_inliers = -1

    print(f"  开始对 {n_points} 个 {point_description} 进行 Sim3 RANSAC (阈值={residual_threshold}m, 迭代={max_trials}, 最小样本={min_samples})...")

    for trial in range(max_trials):
        # 1. 随机选择 min_samples 个点
        indices = np.random.choice(n_points, min_samples, replace=False)
        src_sample = src[indices]
        dst_sample = dst[indices]

        # 2. 用样本点计算一个 Sim3 模型 (R, t, scale)
        R_trial, t_trial, scale_trial = compute_sim3_transform(src_sample, dst_sample)

        if R_trial is None: # 如果样本点无法计算 Sim3 (例如共线)，跳过此次迭代
            continue

        # 3. 用模型变换所有点，计算残差
        src_transformed_trial = scale_trial * (src @ R_trial.T) + t_trial
        residuals = np.linalg.norm(src_transformed_trial - dst, axis=1)

        # 4. 找出内点 (残差小于阈值)
        inlier_mask_trial = residuals < residual_threshold
        num_inliers_trial = np.sum(inlier_mask_trial)

        # 5. 如果当前模型找到更多内点，则更新最佳模型
        if num_inliers_trial > max_inliers:
            max_inliers = num_inliers_trial
            best_inlier_mask = inlier_mask_trial

        # 提前终止条件 (可选，例如找到足够多的内点)
        # if max_inliers >= n_points * stop_probability: # stop_probability 需要定义
        #     break

    print(f"  Sim3 RANSAC 完成: 最多找到 {max_inliers} / {n_points} 个内点。")

    # --- 使用所有找到的最佳内点重新计算 Sim3 ---
    if max_inliers < min_inliers_needed:
        print(f"错误: Sim3 RANSAC: 找到的最佳内点数 ({max_inliers} from {point_description}) 不足所需 ({min_inliers_needed})，无法计算可靠的 Sim3 变换。")
        return None, None, None

    print(f"  使用 {max_inliers} 个内点重新计算最终 Sim3 变换...")
    src_inliers = src[best_inlier_mask]
    dst_inliers = dst[best_inlier_mask]

    final_R, final_t, final_scale = compute_sim3_transform(src_inliers, dst_inliers)

    if final_R is None:
        print(f"错误: Sim3 RANSAC: 使用 {max_inliers} 个内点计算最终变换失败。")
        return None, None, None

    print(f"  最终 Sim3 参数 (基于内点): scale={final_scale:.4f}")
    return final_R, final_t, final_scale

def compute_sim3_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    计算从源点(src)到目标点(dst)的最佳 Sim3 变换 (旋转R, 平移t, 尺度s)。
    使用 Umeyama 算法。
    """
    n_points = src.shape[0]
    if n_points < 3:
         # print(f"错误: 计算 Sim3 变换需要至少 3 个点，但只有 {n_points} 个") # RANSAC 函数会打印更详细信息
         return None, None, None
    if src.shape != dst.shape or src.shape[1] != 3:
         print(f"错误: Sim3 计算：源点或目标点维度不正确 (应为 Nx3)")
         return None, None, None

    try:
        # 1. 计算质心
        src_centroid = np.mean(src, axis=0)
        dst_centroid = np.mean(dst, axis=0)

        # 2. 中心化点集
        src_centered = src - src_centroid
        dst_centered = dst - dst_centroid

        # 3. 计算协方差矩阵 H = sum(pi' * qi'^T)
        H = src_centered.T @ dst_centered

        # 4. SVD 分解 H = U * S * V^T
        U, S, Vt = np.linalg.svd(H)
        V = Vt.T

        # 5. 计算旋转矩阵 R
        R = V @ U.T

        # 特殊情况：处理反射 (确保 R 是旋转矩阵)
        if np.linalg.det(R) < 0:
            # print("警告: 检测到反射（行列式为负），修正旋转矩阵。")
            Vt_copy = Vt.copy()
            Vt_copy[-1, :] *= -1 # 反转 V 的最后一列的符号
            R = Vt_copy.T @ U.T
            # 再次检查行列式
            # if abs(np.linalg.det(R) - 1.0) > 1e-6:
            #     print("错误：修正反射后旋转矩阵行列式仍不为 1。")
            #     return None, None, None

        # 6. 计算尺度 s
        # s = sum(qi' * R * pi') / sum(|pi'|^2)
        src_dist_sq = np.sum(src_centered**2, axis=1)
        var_src = np.sum(src_dist_sq) / n_points # 源点集的方差

        # trace(S * diag(1,1,det(VU^T))) / var_src
        # S 是 H 的奇异值，已经计算得到
        # det(VU^T) 就是 det(R)
        trace_S_diag = np.sum(S) # 如果 det(R)=1
        if np.linalg.det(R) < 0: # 理论上修正后不会发生，但以防万一
            trace_S_diag = S[0] + S[1] - S[2] # 如果 det(R)=-1

        if var_src < 1e-12: # 避免除以零
             print("警告：源点集方差接近零，无法可靠计算尺度，默认为 1.0")
             scale = 1.0
        else:
            scale = trace_S_diag / (n_points * var_src)
            if scale <= 1e-6: # 尺度过小可能表示有问题
                 print(f"警告：计算出的尺度非常小 ({scale:.2e})，可能存在问题。重置为 1.0")
                 scale = 1.0

        # 7. 计算平移 t
        # t = q_centroid - s * R * p_centroid
        t = dst_centroid - scale * (R @ src_centroid)

        return R, t, scale

    except np.linalg.LinAlgError as e:
        print(f"错误: 计算 Sim3 变换时发生线性代数错误 (例如 SVD 失败): {e}")
        return None, None, None
    except Exception as e:
        print(f"错误: 计算 Sim3 变换时发生未知错误: {e}")
        traceback.print_exc()
        return None, None, None


def transform_trajectory(positions: np.ndarray, quaternions: np.ndarray,
                        R: np.ndarray, t: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """应用Sim3变换到整个轨迹（位置和姿态）"""
    # 变换位置: p' = s * R * p + t
    trans_pos = scale * (positions @ R.T) + t # 注意 R 需要转置

    # 变换姿态: q' = R_sim3 * q_orig
    R_sim3_rot = Rotation.from_matrix(R)
    trans_quat_list = []
    for q_xyzw in quaternions:
        original_rot = Rotation.from_quat(q_xyzw)
        new_rot = R_sim3_rot * original_rot # 旋转合成
        new_quat_xyzw = new_rot.as_quat()
        trans_quat_list.append(new_quat_xyzw)

    return trans_pos, np.array(trans_quat_list)

# --- 可视化与评估系统  ---
def plot_results(original_pos: np.ndarray,
                 sim3_pos_ekf_input: np.ndarray,
                 corrected_pos: np.ndarray,
                 gps_pos: np.ndarray,
                 valid_indices_for_error: np.ndarray,
                 aligned_gps_for_error: np.ndarray,
                 slam_times: np.ndarray,
                 ekf_errors: Optional[np.ndarray] = None
                ) -> None:
    """增强的可视化系统，显示原始轨迹、修正后轨迹、GPS点和误差，并提供轨迹显示切换功能"""
    try:
        # 尝试设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    except:
        print("警告：未能设置中文字体 'WenQuanYi Zen Hei' 或 'SimHei'。标签可能显示为方块。请确保已安装相应字体。")

    fig = plt.figure(figsize=(18, 12))
    plt.suptitle("SLAM-GPS 轨迹对齐与融合结果", fontsize=16)

    # 使用 GridSpec 灵活布局
    gs = fig.add_gridspec(2, 3, width_ratios=[0.15, 1, 1], height_ratios=[1,1], wspace=0.3, hspace=0.3)

    # 创建子图区域
    ax_check_buttons = fig.add_subplot(gs[:, 0]) # CheckButtons 占左侧整列
    ax1 = fig.add_subplot(gs[0, 1]) # 2D 轨迹图
    ax3d = fig.add_subplot(gs[0, 2], projection='3d') # 3D 轨迹图
    ax_hist = fig.add_subplot(gs[1, 1]) # 误差直方图
    ax_err_time = fig.add_subplot(gs[1, 2]) # 误差随时间变化图


    # --- 1. 2D轨迹对比 (XY平面) ---
    line_orig, = ax1.plot(original_pos[:, 0], original_pos[:, 1], 'b--', alpha=0.6, linewidth=1, label='原始 SLAM 轨迹')
    line_sim3, = ax1.plot(sim3_pos_ekf_input[:, 0], sim3_pos_ekf_input[:, 1], 'm:', alpha=0.7, linewidth=1, label='Sim3 对齐轨迹 (EKF输入)')
    line_ekf, = ax1.plot(corrected_pos[:, 0], corrected_pos[:, 1], 'g-', linewidth=1.5, label='EKF 融合后轨迹')
    scatter_gps = ax1.scatter(gps_pos[:, 0], gps_pos[:, 1], c='r', marker='.', s=30, label='GPS 参考点 (过滤后, UTM)')

    # 绘制用于误差评估的对齐GPS点 (稀疏显示)
    scatter_aligned_gps = None
    step = max(1, len(valid_indices_for_error) // 100) if len(valid_indices_for_error) > 0 else 1
    if len(valid_indices_for_error) > 0 and aligned_gps_for_error.ndim == 2 and aligned_gps_for_error.shape[0] > 0:
       points_to_scatter = aligned_gps_for_error[::step]
       if points_to_scatter.shape[0] > 0:
           scatter_aligned_gps = ax1.scatter(points_to_scatter[:, 0], points_to_scatter[:, 1],
                       facecolors='none', edgecolors='orange', marker='o', s=40,
                       label='对齐GPS点 (插值, 用于评估)')

    ax1.set_title('轨迹对比 (X-Y 平面)')
    ax1.set_xlabel('X (米)')
    ax1.set_ylabel('Y (米)')
    ax1.grid(True)
    ax1.axis('equal') # 保持 X/Y 轴比例一致
    ax1.legend(loc='best')

    # --- 设置 CheckButtons ---
    lines_to_toggle_map = {
        '原始 SLAM': line_orig,
        'Sim3 对齐': line_sim3,
        'EKF 融合': line_ekf,
        'GPS 参考点': scatter_gps,
        '对齐 GPS (评估)': scatter_aligned_gps, # 可能为 None
    }
    # 过滤掉值为 None 的项 (例如没有对齐的 GPS 点时)
    lines_to_toggle_map = {k: v for k, v in lines_to_toggle_map.items() if v is not None}

    checkbox_labels = list(lines_to_toggle_map.keys())
    initial_visibility = [True] * len(checkbox_labels) # 默认都显示

    check = CheckButtons(
        ax=ax_check_buttons,
        labels=checkbox_labels,
        actives=initial_visibility,
        # 使用列表设置属性值
        label_props={'fontsize': [9]},
        frame_props={'edgecolor': ['gray']},
        check_props={'linewidth': [1.5]}
    )
    ax_check_buttons.set_title("显示/隐藏", fontsize=10) # 缩小标题字号

    # 定义 CheckButtons 的回调函数
    def toggle_line_visibility(label):
        artist = lines_to_toggle_map[label]
        artist.set_visible(not artist.get_visible())
        # 更新图例和画布
        ax1.legend(loc='best') # 更新 2D 图例
        ax3d.legend(loc='best') # 更新 3D 图例
        fig.canvas.draw_idle()

    check.on_clicked(toggle_line_visibility)
    # 将 widget 存储在 figure 对象上，防止被垃圾回收
    fig._widgets_store = [check]


    # --- 2. 3D轨迹对比 ---
    line_orig_3d, = ax3d.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 'b--', alpha=0.6, linewidth=1, label='原始 SLAM 轨迹')
    line_sim3_3d, = ax3d.plot(sim3_pos_ekf_input[:, 0], sim3_pos_ekf_input[:, 1], sim3_pos_ekf_input[:, 2], 'm:', alpha=0.7, linewidth=1, label='Sim3 对齐轨迹 (EKF输入)')
    line_ekf_3d, = ax3d.plot(corrected_pos[:, 0], corrected_pos[:, 1], corrected_pos[:, 2], 'g-', linewidth=1.5, label='EKF 融合后轨迹')
    scatter_gps_3d = ax3d.scatter(gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2], c='r', marker='x', s=30, label='GPS 参考点 (过滤后, UTM)')
    scatter_aligned_gps_3d = None
    if len(valid_indices_for_error) > 0 and aligned_gps_for_error.ndim == 2 and aligned_gps_for_error.shape[0] > 0 :
        points_to_scatter_3d = aligned_gps_for_error[::step]
        if points_to_scatter_3d.shape[0] > 0:
            scatter_aligned_gps_3d = ax3d.scatter(points_to_scatter_3d[:, 0], points_to_scatter_3d[:, 1], points_to_scatter_3d[:, 2],
                        facecolors='none', edgecolors='orange', marker='o', s=40,
                        label='对齐GPS点 (插值, 用于评估)')

    ax3d.set_title('轨迹对比 (3D)')
    ax3d.set_xlabel('X (米)')
    ax3d.set_ylabel('Y (米)')
    ax3d.set_zlabel('Z (米)')
    ax3d.legend(loc='best')

    # 将 3D 图的 CheckButtons 控制也连接起来
    lines_to_toggle_map_3d = {
        '原始 SLAM': line_orig_3d,
        'Sim3 对齐': line_sim3_3d,
        'EKF 融合': line_ekf_3d,
        'GPS 参考点': scatter_gps_3d,
        '对齐 GPS (评估)': scatter_aligned_gps_3d, # 可能为 None
    }
    lines_to_toggle_map_3d = {k: v for k, v in lines_to_toggle_map_3d.items() if v is not None}

    # 更新回调函数以同时控制 2D 和 3D 图
    def toggle_line_visibility_combined(label):
        artist_2d = lines_to_toggle_map.get(label)
        artist_3d = lines_to_toggle_map_3d.get(label)
        current_visibility = None
        if artist_2d:
            current_visibility = artist_2d.get_visible()
            artist_2d.set_visible(not current_visibility)
        if artist_3d:
            # 如果 2D 不存在，则以 3D 的当前状态为准
            if current_visibility is None:
                current_visibility = artist_3d.get_visible()
            artist_3d.set_visible(not current_visibility)

        ax1.legend(loc='best')
        ax3d.legend(loc='best')
        fig.canvas.draw_idle()

    # 重新绑定回调函数
    check.disconnect_events() # 断开旧的回调
    check.on_clicked(toggle_line_visibility_combined) # 绑定新的回调

    # 尝试自动调整 3D 视图范围
    try:
        # 收集所有有效的轨迹点用于计算范围
        all_traj_points = [p for p in [original_pos, sim3_pos_ekf_input, corrected_pos, gps_pos] if p.ndim == 2 and p.shape[0] > 0]
        if not all_traj_points:
             plot_center_data = np.array([[0,0,0]]) # 如果没有数据，设置默认中心
        else:
            plot_center_data = np.vstack(all_traj_points)

        # 计算最大范围的一半，并增加一点边距
        max_range = np.array([plot_center_data[:,0].max()-plot_center_data[:,0].min(),
                              plot_center_data[:,1].max()-plot_center_data[:,1].min(),
                              plot_center_data[:,2].max()-plot_center_data[:,2].min()]).max() / 2.0 * 1.1
        if max_range < 1.0: max_range = 5.0 # 避免范围过小

        # 优先使用 EKF 轨迹计算中心点
        center_priority = [corrected_pos, sim3_pos_ekf_input, original_pos, gps_pos]
        final_center_data = None
        for cp_data in center_priority:
            if cp_data is not None and cp_data.ndim == 2 and cp_data.shape[0] > 0:
                final_center_data = cp_data
                break
        if final_center_data is None: final_center_data = plot_center_data # 如果都没有，使用所有点

        # 计算中位数作为中心点（比均值更稳健）
        mid_x = np.median(final_center_data[:,0])
        mid_y = np.median(final_center_data[:,1])
        mid_z = np.median(final_center_data[:,2])
        ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    except Exception as e_3d:
        print(f"警告 (plot_results): 3D 图自动缩放失败: {e_3d}")
        pass # 缩放失败不影响程序运行

    # --- 3. 最终误差分布 (直方图) ---
    if ekf_errors is not None and len(ekf_errors) > 0:
        mean_err = np.mean(ekf_errors)
        std_err = np.std(ekf_errors)
        max_err = np.max(ekf_errors)
        median_err = np.median(ekf_errors)
        rmse = np.sqrt(np.mean(ekf_errors**2)) # 计算 RMSE

        ax_hist.hist(ekf_errors, bins=30, alpha=0.75, color='purple', label='误差分布')
        ax_hist.axvline(mean_err, color='red', linestyle='dashed', linewidth=1, label=f'均值: {mean_err:.2f}m')
        ax_hist.axvline(median_err, color='orange', linestyle='dashed', linewidth=1, label=f'中位数: {median_err:.2f}m')
        ax_hist.axvline(rmse, color='cyan', linestyle='dotted', linewidth=1, label=f'RMSE: {rmse:.2f}m')

        ax_hist.set_title(f'最终位置误差分布 (N={len(ekf_errors)})\n标准差: {std_err:.2f}m, 最大值: {max_err:.2f}m')
        ax_hist.set_xlabel('绝对位置误差 (米)')
        ax_hist.set_ylabel('频数')
        ax_hist.legend()
        ax_hist.grid(axis='y', linestyle=':') # 只显示 y 轴网格
    else:
        ax_hist.set_title("最终位置误差分布")
        ax_hist.text(0.5, 0.5, "无有效误差数据", ha='center', va='center', fontsize=12, transform=ax_hist.transAxes)
        ax_hist.set_xlabel('误差 (米)')
        ax_hist.set_ylabel('频数')


    # --- 4. 误差随时间变化图 ---
    if ekf_errors is not None and len(valid_indices_for_error) > 0 and len(slam_times) > 0:
        # 确保时间戳和误差数据长度一致
        if corrected_pos.shape[0] == len(slam_times):
            valid_timestamps = slam_times[valid_indices_for_error]
            if len(valid_timestamps) == len(ekf_errors) and len(ekf_errors) > 0 :
                 relative_time = valid_timestamps - valid_timestamps[0] # 计算相对时间
                 ax_err_time.plot(relative_time, ekf_errors, 'r-', linewidth=1, alpha=0.8, label='绝对位置误差')
                 ax_err_time.set_xlabel('相对时间 (秒)')
                 ax_err_time.set_ylabel('误差 (米)')
                 ax_err_time.set_title('误差随时间变化')
                 ax_err_time.grid(True)
                 ax_err_time.legend()
                 # 调整 Y 轴下限，避免从 0 开始导致曲线看不清
                 min_plot_err = 0
                 if len(ekf_errors) > 1 and np.max(ekf_errors) > np.min(ekf_errors):
                     min_plot_err = max(0, np.min(ekf_errors) - 0.1 * (np.max(ekf_errors) - np.min(ekf_errors)))
                 elif len(ekf_errors) == 1:
                     min_plot_err = max(0, ekf_errors[0] * 0.9)
                 ax_err_time.set_ylim(bottom=min_plot_err)
            else:
                 # 如果时间戳和误差数量不匹配
                 ax_err_time.set_title("误差随时间变化")
                 ax_err_time.text(0.5, 0.5, "时间戳与误差数量不匹配", ha='center', va='center', fontsize=10, transform=ax_err_time.transAxes)
                 ax_err_time.set_xlabel('相对时间 (秒)')
                 ax_err_time.set_ylabel('误差 (米)')
        else:
            # 如果 SLAM 时间戳和位姿数量不匹配
            ax_err_time.set_title("误差随时间变化")
            ax_err_time.text(0.5, 0.5, "SLAM 时间戳与位姿数量不匹配", ha='center', va='center', fontsize=10, transform=ax_err_time.transAxes)
            ax_err_time.set_xlabel('相对时间 (秒)')
            ax_err_time.set_ylabel('误差 (米)')
    else:
         # 如果没有误差数据
         ax_err_time.set_title("误差随时间变化")
         ax_err_time.text(0.5, 0.5, "无有效误差数据或时间戳", ha='center', va='center', fontsize=10, transform=ax_err_time.transAxes)
         ax_err_time.set_xlabel('相对时间 (秒)')
         ax_err_time.set_ylabel('误差 (米)')

    # 调整整体布局，防止标题重叠
    fig.tight_layout(rect=[0.08, 0.03, 1, 0.95]) # 调整左边距给 CheckButtons 留空间
    fig.subplots_adjust(top=0.92) # 调整顶部边距给总标题

    plt.show()


# --- GUI文件选择功能 ---
def select_file_dialog(title: str, filetypes: List[Tuple[str, str]]) -> str:
    """通用文件选择对话框"""
    root = tk.Tk()
    root.withdraw() # 隐藏主窗口
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy() # 关闭 tk 实例
    return file_path if file_path else ""

def select_slam_file() -> str:
    return select_file_dialog(
        "选择 SLAM 轨迹文件 (TUM格式: timestamp tx ty tz qx qy qz qw)",
        [("文本文件", "*.txt"), ("所有文件", "*.*")]
    )

def select_gps_file() -> str:
    return select_file_dialog(
        "选择 GPS 数据文件 (格式: timestamp lat lon alt [...], 空格或逗号分隔)",
        [("文本文件", "*.txt"), ("CSV文件", "*.csv"),("所有文件", "*.*")]
    )

# ----------------------------
# EKF 实现 
# ----------------------------

class ExtendedKalmanFilter:
    def __init__(self, initial_pos: np.ndarray, initial_quat: np.ndarray,
                 config: Dict[str, Any]): # 传入完整的 EKF 配置字典
        """
        初始化扩展卡尔曼滤波器 (实现航位推算+平滑恢复逻辑)
        状态向量 (state): [x, y, z, qx, qy, qz, qw] (7维)
        """
        if not (initial_pos.shape == (3,) and initial_quat.shape == (4,)):
            raise ValueError(f"EKF 初始化: 初始位置或四元数维度错误")

        ekf_config = config # 直接使用传入的 EKF 子配置

        initial_quat_normalized = self.normalize_quaternion(initial_quat)
        self.state = np.concatenate([initial_pos, initial_quat_normalized]).astype(float)
        self.cov = np.diag(ekf_config['initial_cov_diag']).astype(float)
        self.Q_per_sec = np.diag(ekf_config['process_noise_diag']).astype(float) # 每秒的过程噪声
        self.R = np.diag(ekf_config['meas_noise_diag']).astype(float) # 测量噪声

        # 检查维度以防配置错误
        if self.state.shape != (7,): raise ValueError(f"EKF: 内部状态维度错误 (应为 7)")
        if self.cov.shape != (7, 7): raise ValueError(f"EKF: 内部协方差维度错误 (应为 7x7)")
        if self.Q_per_sec.shape != (7, 7): raise ValueError(f"EKF: 内部过程噪声Q维度错误 (应为 7x7)")
        if self.R.shape != (3, 3): raise ValueError(f"EKF: 内部测量噪声R维度错误 (应为 3x3)")

        # --- 新增: 平滑相关状态 ---
        self.gnss_available_prev = None # 上一时刻的 GNSS 可用状态
        self.gnss_update_weight = 0.0 # GNSS 更新贡献的权重 (0.0 到 1.0)
        self.transition_steps = max(1, int(ekf_config.get('transition_steps', 10))) # 从配置读取平滑步数
        self.weight_delta = 1.0 / self.transition_steps if self.transition_steps > 0 else 1.0 # 每步权重增加量
        self._last_predicted_state = self.state.copy() # 存储上一步的预测结果，用于平滑插值

    @staticmethod
    def normalize_quaternion(q: np.ndarray) -> np.ndarray:
        """归一化四元数"""
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            print("警告：尝试归一化零四元数，返回默认单位四元数 [0,0,0,1]。")
            return np.array([0.0, 0.0, 0.0, 1.0])
        return q / norm

    def _predict(self, slam_motion_update: Tuple[np.ndarray, np.ndarray], delta_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF 预测步骤：使用 SLAM 相对运动更新状态和协方差。
        Args:
            slam_motion_update: (delta_pos_local, delta_quat) 来自 calculate_relative_pose
            delta_time: 时间间隔
        Returns:
            predicted_state, predicted_covariance
        """
        # --- 1. 获取上一时刻状态 ---
        prev_state = self.state
        prev_cov = self.cov
        prev_pos = prev_state[:3]
        prev_quat = prev_state[3:] # [x, y, z, w]
        prev_rot = Rotation.from_quat(prev_quat)

        delta_pos_local, delta_quat = slam_motion_update
        delta_rot = Rotation.from_quat(delta_quat)

        # --- 2. 状态预测 (运动模型) ---
        # 位置预测: pos_pred = pos_prev + R(q_prev) @ delta_pos_local
        predicted_pos = prev_pos + prev_rot.apply(delta_pos_local)
        # 姿态预测: q_pred = q_prev * delta_q (四元数乘法)
        predicted_rot = prev_rot * delta_rot
        predicted_quat = self.normalize_quaternion(predicted_rot.as_quat()) # 预测后归一化 [x, y, z, w]

        predicted_state = np.concatenate([predicted_pos, predicted_quat])

        # --- 3. 计算状态转移矩阵 Jacobian F (7x7) ---
        # F = ∂f/∂x | evaluated at prev_state, motion_update
        # 对于7D状态（位置+四元数），精确的雅可比计算非常复杂。
        # F 的非对角线块（∂pos/∂quat 和 ∂quat/∂pos）通常难以精确计算或影响较小。
        # 常见的简化是假设 F 为单位矩阵，或者只考虑旋转对位置预测的影响。
        # 这里采用简化的 F = 单位矩阵，将模型的不确定性更多地归因于过程噪声 Q。
        F = np.eye(7)
        # (如果需要更高精度，可以推导更复杂的 F 矩阵)

        # --- 4. 调整过程噪声 Q for delta_time ---
        # 假设过程噪声与时间间隔成正比
        dt_adjusted = max(abs(delta_time), 1e-6) # 确保 dt 为正
        Q = self.Q_per_sec * dt_adjusted # 根据时间间隔缩放过程噪声协方差

        # --- 5. 预测协方差 ---
        # P_pred = F * P_prev * F^T + Q
        # 使用 F = eye(7) 简化为 P_pred = P_prev + Q
        predicted_covariance = prev_cov + Q
        # 确保协方差矩阵对称性
        predicted_covariance = (predicted_covariance + predicted_covariance.T) / 2.0

        # 缓存预测结果，用于后续可能的平滑处理
        self._last_predicted_state = predicted_state.copy()

        return predicted_state, predicted_covariance

    def _update(self, predicted_state: np.ndarray, predicted_covariance: np.ndarray,
                gps_pos: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        EKF 更新步骤：使用 GPS 位置测量修正状态估计。
        Args:
            predicted_state, predicted_covariance: 来自 _predict 步骤
            gps_pos: GPS 测量值 [x, y, z]
        Returns:
            (updated_state, updated_covariance) 或 (None, None) 如果更新失败
        """
        # 检查 GPS 测量值是否有效
        if gps_pos.shape != (3,) or np.isnan(gps_pos).any():
            # print("警告 (EKF Update): 无效的 GPS 测量值，跳过更新。")
            return None, None # 无效测量，不更新

        # 测量矩阵 H (观测模型)：只观测位置 x, y, z
        # H 将 7D 状态映射到 3D 测量空间
        H = np.zeros((3, 7))
        H[0, 0] = 1.0 # x_meas = x_state
        H[1, 1] = 1.0 # y_meas = y_state
        H[2, 2] = 1.0 # z_meas = z_state

        try:
            # 1. 计算测量残差 (Innovation): y = z_gps - h(x_predicted)
            # 在这里，h(x_predicted) = H * x_predicted，即预测状态的位置部分
            predicted_measurement = predicted_state[:3]
            innovation = gps_pos - predicted_measurement

            # 2. 计算残差协方差 (Innovation Covariance): S = H * P_pred * H^T + R
            H_P = H @ predicted_covariance # (3x7) @ (7x7) = (3x7)
            S = H_P @ H.T + self.R         # (3x7) @ (7x3) + (3x3) = (3x3)
            # 确保 S 对称
            S = (S + S.T) / 2.0

            # 检查 S 是否奇异或病态 (行列式接近0)
            if abs(np.linalg.det(S)) < 1e-12:
                print(f"警告 (EKF Update): S 矩阵接近奇异 (det={np.linalg.det(S):.2e})，跳过更新。可能是测量噪声 R 设置过小或协方差 P 崩溃。")
                return None, None

            # 3. 计算卡尔曼增益 (Kalman Gain): K = P_pred * H^T * S^(-1)
            S_inv = np.linalg.inv(S)           # (3x3)
            K = predicted_covariance @ H.T @ S_inv # (7x7) @ (7x3) @ (3x3) = (7x3)

            # 4. 状态更新: x_new = x_predicted + K * y
            updated_state = predicted_state + K @ innovation # (7,) + (7x3) @ (3,) = (7,)
            # 更新后必须重新归一化四元数部分
            updated_state[3:] = self.normalize_quaternion(updated_state[3:])

            # 5. 协方差更新: P_new = (I - K * H) * P_predicted (标准形式)
            # 或者使用 Joseph 形式（数值稳定性更好）:
            # P_new = (I - K @ H) @ P_predicted @ (I - K @ H).T + K @ R @ K.T
            I = np.eye(7)
            # P_new = (I - K @ H) @ predicted_covariance # 标准形式
            P_new = (I - K @ H) @ predicted_covariance @ (I - K @ H).T + K @ self.R @ K.T # Joseph 形式
            # 再次确保对称性
            updated_covariance = (P_new + P_new.T) / 2.0

            return updated_state, updated_covariance

        except np.linalg.LinAlgError as e:
            print(f"警告 (EKF Update): 更新步骤中发生线性代数错误 (例如求逆失败): {e}。跳过更新。")
            return None, None
        except Exception as e:
            print(f"警告 (EKF Update): 更新步骤中发生未知错误: {e}。跳过更新。")
            traceback.print_exc()
            return None, None

    def process_step(self, slam_motion_update: Tuple[np.ndarray, np.ndarray],
                     gps_measurement: Optional[np.ndarray], gnss_available: bool,
                     delta_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理一个时间步：执行预测、(可选)更新、(可选)平滑，并更新内部状态。
        Args:
            slam_motion_update: SLAM 提供的相对运动 (delta_pos_local, delta_quat)
            gps_measurement: 当前时间点的 GPS 测量值 (x, y, z)，如果不可用则为 None
            gnss_available: 当前时间点 GNSS 是否有效
            delta_time: 距离上一步的时间间隔

        Returns:
            final_fused_pose (pos, quat): 当前步骤的最终融合位姿估计。
        """
        # --- 1. EKF 预测 ---
        # 使用 SLAM 运动信息预测下一状态和协方差
        predicted_state, predicted_covariance = self._predict(slam_motion_update, delta_time)

        # --- 2. EKF 更新 (如果GNSS可用且测量值有效) ---
        updated_state = None
        updated_covariance = None
        update_successful = False # 标记更新是否成功执行
        if gnss_available and gps_measurement is not None:
            res = self._update(predicted_state, predicted_covariance, gps_measurement)
            if res[0] is not None: # 检查 _update 是否返回了有效结果
                updated_state, updated_covariance = res
                update_successful = True # 标记更新成功

        # --- 3. 处理GNSS状态变化和平滑权重 ---
        # 检测 GNSS 是否刚刚从不可用变为可用
        just_recovered = gnss_available and (self.gnss_available_prev == False)

        if gnss_available: # 如果当前 GNSS 可用
            if just_recovered:
                # 如果是刚恢复，重置权重为起始值
                self.gnss_update_weight = self.weight_delta
                # print(f"  [EKF Smooth] GNSS 恢复，开始平滑过渡 (权重={self.gnss_update_weight:.2f})")
            elif self.gnss_update_weight < 1.0:
                # 如果仍在平滑过渡中，增加权重
                self.gnss_update_weight = min(1.0, self.gnss_update_weight + self.weight_delta)
            # else: 平滑已完成，权重保持 1.0
        else: # 如果当前 GNSS 不可用
            if self.gnss_available_prev == True: # 如果是从可用变为不可用
                 # print("  [EKF Smooth] GNSS 丢失，停止平滑。")
                 pass
            self.gnss_update_weight = 0.0 # 重置权重

        # --- 4. 计算最终融合状态 (应用平滑) ---
        # 默认使用预测结果
        final_fused_state = predicted_state
        final_fused_covariance = predicted_covariance

        # 平滑条件：GNSS 可用、更新成功、且权重小于 1.0 (即处于过渡阶段)
        if gnss_available and update_successful and self.gnss_update_weight < 1.0:
            w = self.gnss_update_weight # 当前更新结果的权重
            # 使用上一步缓存的预测状态进行插值
            pred_state_for_smooth = self._last_predicted_state

            # 位置线性插值
            smooth_pos = (1.0 - w) * pred_state_for_smooth[:3] + w * updated_state[:3]
            # 姿态球面线性插值 (NLERP)
            smooth_quat = quaternion_nlerp(pred_state_for_smooth[3:], updated_state[3:], w)

            final_fused_state = np.concatenate([smooth_pos, smooth_quat])
            # 平滑期间的协方差：简单地使用更新后的协方差 (代表了融合后的更优估计)
            final_fused_covariance = updated_covariance
            # print(f"    平滑中... 权重={w:.2f}")

        # 如果 GNSS 可用且更新成功，并且平滑已完成 (w=1.0) 或未启用平滑，则直接使用更新结果
        elif gnss_available and update_successful:
             final_fused_state = updated_state
             final_fused_covariance = updated_covariance

        # 如果 GNSS 不可用，或更新失败，则使用预测结果 (已在初始化时设置)

        # --- 5. 更新EKF内部状态以备下一步使用 ---
        self.state = final_fused_state.copy()
        self.cov = final_fused_covariance.copy()

        # --- 6. 记录当前 GNSS 状态，供下一步判断是否恢复 ---
        self.gnss_available_prev = gnss_available

        # --- 7. 返回当前步最终融合结果 ---
        return self.state[:3].copy(), self.state[3:].copy() # 返回位置和四元数

# ----------------------------
# EKF 轨迹修正应用函数
# ----------------------------

def apply_ekf_correction(slam_data: Dict[str, np.ndarray], gps_data: Dict[str, np.ndarray],
                         sim3_pos: np.ndarray, sim3_quat: np.ndarray,
                         config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用 EKF 对经过 Sim3 全局变换的轨迹进行逐点融合修正。
    使用包含平滑逻辑的 EKF 类。
    """
    n_points = len(slam_data['timestamps'])
    if n_points == 0:
        print("错误 (apply_ekf_correction): 输入的 SLAM 数据为空。")
        return np.empty((0, 3)), np.empty((0, 4))
    if not (sim3_pos.shape[0] == n_points and sim3_quat.shape[0] == n_points):
         raise ValueError(f"错误: Sim3 变换后的轨迹点数 ({sim3_pos.shape[0]}) 与原始 SLAM 时间戳数量 ({n_points}) 不匹配。")

    ekf_config = config['ekf'] # 获取 EKF 配置
    time_align_config = config['time_alignment'] # 获取时间对齐配置

    # 1. 初始化 EKF (传入 EKF 相关配置)
    try:
        ekf = ExtendedKalmanFilter(
            initial_pos=sim3_pos[0],    # 使用 Sim3 变换后的第一个点初始化
            initial_quat=sim3_quat[0],
            config=ekf_config           # 传入 EKF 子配置字典
        )
        # 为了正确设置 EKF 的初始 gnss_available_prev 状态，需要检查第一个点是否有对应的 GPS
        # 我们需要重新做一次时间对齐来获取第一个点的有效性 (虽然有点冗余，但逻辑最清晰)
        aligned_gps_for_init, valid_mask_for_init = dynamic_time_alignment(
            slam_data, gps_data, time_align_config
        )
        # 设置 EKF 的初始 previous GNSS 状态
        ekf.gnss_available_prev = valid_mask_for_init[0] if n_points > 0 and len(valid_mask_for_init) > 0 else False
        print(f"EKF 初始化完成。初始 GNSS 状态 (上一时刻): {ekf.gnss_available_prev}")

    except Exception as e:
         raise ValueError(f"EKF 初始化失败: {e}")

    # 2. 获取对齐的 GPS 数据用于 EKF 更新
    # 直接使用上面初始化时计算的对齐结果
    print("使用时间对齐结果获取 EKF 更新所需的 GPS 测量...")
    aligned_gps_for_update = aligned_gps_for_init
    valid_mask_for_update = valid_mask_for_init
    num_valid_gps_for_update = np.sum(valid_mask_for_update)
    print(f"EKF 修正：共有 {num_valid_gps_for_update} / {n_points} 个时间点将尝试使用有效的 GPS 测量进行更新。")

    # 准备存储修正后的轨迹
    corrected_pos_list = np.zeros_like(sim3_pos)
    corrected_quat_list = np.zeros_like(sim3_quat)

    # 第一个点的状态就是 EKF 的初始状态
    corrected_pos_list[0], corrected_quat_list[0] = ekf.state[:3].copy(), ekf.state[3:].copy()

    # --- 获取 *原始* SLAM 位姿用于计算相对运动 ---
    # EKF 的预测步骤需要基于原始 SLAM 的相对运动量
    orig_slam_pos = slam_data['positions']
    orig_slam_quat = slam_data['quaternions']

    last_time = slam_data['timestamps'][0]
    predict_steps = 0
    updates_attempted = 0 # 尝试更新次数 (有GPS测量)
    updates_successful = 0 # 成功更新次数 (EKF内部返回了结果)

    # 3. 迭代处理每个 SLAM 时间点 (从第二个点开始)
    for i in range(1, n_points):
        current_time = slam_data['timestamps'][i]
        delta_t = current_time - last_time
        if delta_t <= 1e-9: # 避免零或负 dt
            # print(f"警告 (EKF loop): 时间戳间隔过小或非单调在索引 {i} (dt={delta_t:.4f}s)。使用小正值。")
            delta_t = 1e-6 # 使用一个非常小的正时间间隔，Q 影响会很小

        # a) 计算 SLAM 的相对运动 (Motion Update)
        # 使用 *原始* SLAM 数据计算相邻帧之间的相对运动
        slam_motion_update = calculate_relative_pose(
            orig_slam_pos[i-1], orig_slam_quat[i-1],
            orig_slam_pos[i], orig_slam_quat[i]
        )

        # b) 获取当前时间点的 GPS 测量和可用性
        gnss_available = valid_mask_for_update[i] if i < len(valid_mask_for_update) else False
        gps_measurement = aligned_gps_for_update[i] if gnss_available and i < len(aligned_gps_for_update) else None
        # 再次检查插值结果是否为 NaN
        if gps_measurement is not None and np.isnan(gps_measurement).any():
            gnss_available = False
            gps_measurement = None

        # c) 调用 EKF 的处理步骤
        # process_step 内部会处理预测、更新、平滑，并更新 EKF 的内部状态
        prev_state_before_step = ekf.state.copy() # 记录一下处理前的状态
        fused_pos, fused_quat = ekf.process_step(
            slam_motion_update=slam_motion_update,
            gps_measurement=gps_measurement,
            gnss_available=gnss_available,
            delta_time=delta_t
        )
        predict_steps += 1
        if gnss_available:
             updates_attempted += 1
             # 检查状态是否真的被更新了 (与预测结果不同)
             # 注意：这里不能简单比较，因为平滑也会改变状态
             # 需要在 EKF 内部增加一个标志位来判断更新是否成功执行并返回了结果

        # d) 记录当前 EKF 修正后的状态
        corrected_pos_list[i] = fused_pos
        corrected_quat_list[i] = fused_quat
        last_time = current_time

    print(f"EKF 修正完成。共执行 {predict_steps} 步处理。")
    print(f"  尝试进行 GPS 更新的步数: {updates_attempted} / {predict_steps}")
    # (需要修改 EKF 类以获取更准确的成功更新次数)

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
        gps_data = load_gps_data(gps_path) # 内部处理投影和过滤
        print(f"  SLAM 点数: {len(slam_data['positions'])}")
        print(f"  GPS 点数 (有效、投影并过滤后): {len(gps_data['positions'])}")
        if len(slam_data['positions']) == 0 or len(gps_data['positions']) < 2:
             raise ValueError("SLAM 数据为空 或 有效 GPS 数据点不足 (<2)，无法继续处理。")
        print("数据加载与预处理完成。")
        print("-" * 30)

        # 3. 时间对齐 (获取用于 Sim3 的匹配点)
        print("步骤 2/7: 执行时间对齐以获取 Sim3 匹配点...")
        aligned_gps_for_sim3, valid_mask_sim3 = dynamic_time_alignment(
            slam_data, gps_data, CONFIG['time_alignment']
        )
        # 获取所有有效匹配点的索引
        valid_indices_all = np.where(valid_mask_sim3)[0]
        min_points_needed_for_sim3 = CONFIG['sim3_ransac']['min_inliers_needed'] # RANSAC 至少需要的内点数
        min_samples_for_sim3 = CONFIG['sim3_ransac']['min_samples'] # RANSAC 每次迭代需要的样本数

        if len(valid_indices_all) < min_samples_for_sim3: # 首先检查总点数是否够 RANSAC 抽样
            raise ValueError(f"有效时间同步匹配点总数不足 ({len(valid_indices_all)} < {min_samples_for_sim3})，无法进行 Sim3 变换估计。")

        print(f"找到 {len(valid_indices_all)} 个有效时间同步的匹配点。")

        # === 新增：识别第一段连续匹配并应用时间阈值 ===
        print("  正在识别第一段连续匹配...")
        sim3_calc_indices = np.array([], dtype=int) # 初始化用于计算 Sim3 的索引数组
        point_description_for_ransac = "all valid points" # RANSAC 打印信息

        if len(valid_indices_all) > 0:
            valid_slam_times = slam_data['timestamps'][valid_indices_all]
            time_diffs = np.diff(valid_slam_times)
            max_gap = CONFIG['time_alignment']['max_gps_gap_threshold']
            # 找到第一个时间差大于阈值的索引
            first_gap_index = np.where(time_diffs > max_gap)[0]

            end_of_first_segment_idx = len(valid_indices_all) # 默认使用所有点
            if len(first_gap_index) > 0:
                end_of_first_segment_idx = first_gap_index[0] + 1 # 第一个间隙之前的点属于第一段
                print(f"  第一段连续匹配结束于索引 {end_of_first_segment_idx} (共 {end_of_first_segment_idx} 个点), 因时间间隙 > {max_gap:.1f}s。")
            else:
                print(f"  未发现明显时间间隙，第一段连续匹配包含所有 {len(valid_indices_all)} 个有效点。")

            # 获取第一段连续匹配的原始索引
            first_segment_indices = valid_indices_all[:end_of_first_segment_idx]

            if len(first_segment_indices) < min_samples_for_sim3:
                print(f"警告：第一段连续匹配点数 ({len(first_segment_indices)}) 少于 RANSAC 最小样本数 ({min_samples_for_sim3})。将尝试使用所有有效点。")
                sim3_calc_indices = valid_indices_all # 回退到使用所有点
                point_description_for_ransac = "all valid points (first segment too short)"
            else:
                # 应用时间阈值
                max_duration = CONFIG['sim3_ransac']['max_initial_duration']
                first_segment_slam_times = slam_data['timestamps'][first_segment_indices]
                segment_start_time = first_segment_slam_times[0]
                time_limited_mask = (first_segment_slam_times <= segment_start_time + max_duration)
                sim3_calc_indices_timed = first_segment_indices[time_limited_mask]
                num_timed = len(sim3_calc_indices_timed)

                print(f"  应用最大时长阈值 {max_duration:.1f}s: 从第一段的 {len(first_segment_indices)} 个点中选取了 {num_timed} 个点。")

                # 检查应用阈值后点数是否足够 RANSAC 抽样
                if num_timed < min_samples_for_sim3:
                    print(f"警告：应用时间阈值后点数 ({num_timed}) 少于 RANSAC 最小样本数 ({min_samples_for_sim3})。将使用完整的第一段 ({len(first_segment_indices)} 个点)。")
                    sim3_calc_indices = first_segment_indices
                    point_description_for_ransac = f"first segment ({len(first_segment_indices)} points, time threshold removed)"
                else:
                    sim3_calc_indices = sim3_calc_indices_timed
                    point_description_for_ransac = f"initial segment (up to {max_duration:.1f}s, {num_timed} points)"
        else:
            # 理论上不会执行到这里，因为前面已经检查过 len(valid_indices_all)
             raise ValueError("代码逻辑错误：valid_indices_all 为空但未被捕获。")

        # 最终检查用于计算 Sim3 的点数是否足够 RANSAC 抽样
        if len(sim3_calc_indices) < min_samples_for_sim3:
             raise ValueError(f"最终用于 Sim3 计算的点数不足 ({len(sim3_calc_indices)} < {min_samples_for_sim3})，无法继续。")

        print(f"最终将使用 {len(sim3_calc_indices)} 个点 ({point_description_for_ransac}) 来计算 Sim3 变换。")
        # === 结束：识别第一段并应用阈值 ===

        print("时间对齐与 Sim3 点选择完成。")
        print("-" * 30)

        # 4. Sim3 全局对齐 (使用选定的点集)
        print("步骤 3/7: 计算稳健的 Sim3 全局变换...")
        # 使用 sim3_calc_indices 来选取源点和目标点
        src_points = slam_data['positions'][sim3_calc_indices]
        dst_points = aligned_gps_for_sim3[sim3_calc_indices]

        R, t, scale = compute_sim3_transform_robust(
            src=src_points,
            dst=dst_points,
            min_samples=min_samples_for_sim3, # 使用 RANSAC 配置
            residual_threshold=CONFIG['sim3_ransac']['residual_threshold'],
            max_trials=CONFIG['sim3_ransac']['max_trials'],
            min_inliers_needed=min_points_needed_for_sim3, # 使用 RANSAC 配置
            point_description=point_description_for_ransac # 传递点的描述信息
        )
        if R is None or t is None or scale is None:
             raise RuntimeError(f"Sim3 全局变换计算失败 (基于 {point_description_for_ransac})，无法继续。")

        print(f"Sim3 全局变换计算成功 (基于 {point_description_for_ransac})。")
        print("步骤 4/7: 应用 Sim3 变换到整个 SLAM 轨迹...")
        # transform_trajectory 总是对整个轨迹应用变换
        sim3_pos, sim3_quat = transform_trajectory(
            slam_data['positions'], slam_data['quaternions'], R, t, scale
        )
        print("Sim3 变换应用完成。")
        print("-" * 30)

        # 5. EKF 局部修正 (融合) - 调用修改后的函数
        print("步骤 5/7: 应用 EKF 进行轨迹融合与修正...")
        corrected_pos, corrected_quat = apply_ekf_correction(
            slam_data, gps_data, sim3_pos, sim3_quat, CONFIG # 传递完整配置
        )
        print("EKF 轨迹融合与修正完成。")
        print("-" * 30)

        # 6. 评估误差
        print("步骤 6/7: 评估轨迹误差...")
        # 重新进行一次时间对齐以获取用于评估的、未被 Sim3 影响的对齐点
        # （注意：这里假设评估也应该基于所有有效的对齐点，而不是仅基于第一段）
        aligned_gps_for_eval, valid_mask_eval = dynamic_time_alignment(
             slam_data, gps_data, CONFIG['time_alignment']
        )
        valid_indices_eval = np.where(valid_mask_eval)[0]
        eval_gps_points = np.empty((0,3)) # 初始化空的评估用 GPS 点
        ekf_errors = None # 初始化 EKF 误差

        if len(valid_indices_eval) > 0:
            eval_gps_points = aligned_gps_for_eval[valid_indices_eval]
            print(f"  (基于 {len(valid_indices_eval)} 个有效对齐点进行评估)")

            # 原始 SLAM vs 对齐 GPS
            raw_positions_eval = slam_data['positions'][valid_indices_eval]
            raw_errors = np.linalg.norm(raw_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] 原始轨迹   vs 对齐GPS -> 均值误差: {np.mean(raw_errors):.3f} m, 中位数误差: {np.median(raw_errors):.3f} m, RMSE: {np.sqrt(np.mean(raw_errors**2)):.3f} m")

            # Sim3 对齐后 vs 对齐 GPS
            sim3_positions_eval = sim3_pos[valid_indices_eval]
            sim3_errors = np.linalg.norm(sim3_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] Sim3 对齐后 vs 对齐GPS -> 均值误差: {np.mean(sim3_errors):.3f} m, 中位数误差: {np.median(sim3_errors):.3f} m, RMSE: {np.sqrt(np.mean(sim3_errors**2)):.3f} m")

            # EKF 融合后 vs 对齐 GPS
            ekf_positions_eval = corrected_pos[valid_indices_eval]
            ekf_errors = np.linalg.norm(ekf_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] EKF 融合后  vs 对齐GPS -> 均值误差: {np.mean(ekf_errors):.3f} m, 中位数误差: {np.median(ekf_errors):.3f} m, RMSE: {np.sqrt(np.mean(ekf_errors**2)):.3f} m")
        else:
            print("警告：无有效对齐的GPS点用于最终误差评估。")
            ekf_errors = None # 确保为 None
        print("误差评估完成。")
        print("-" * 30)

        # 7. 保存结果
        print("步骤 7/7: 保存结果与可视化...")
        save_results = messagebox.askyesno("保存结果", "处理完成。是否要保存修正后的轨迹?")
        if save_results:
            # 提取原始 SLAM 文件名作为默认保存名的基础
            base_filename = slam_path.split('/')[-1].split('\\')[-1]
            default_save_name = base_filename.replace('.txt', '_corrected_utm.txt')
            # 如果文件名没有 .txt 后缀，则添加后缀
            if default_save_name == base_filename: default_save_name += '_corrected_utm.txt'

            # 保存 UTM 坐标轨迹
            output_path_utm = filedialog.asksaveasfilename(
                title="保存修正后的轨迹 (UTM 坐标, TUM 格式)", defaultextension=".txt",
                filetypes=[("文本文件", "*.txt")], initialfile=default_save_name
            )
            if output_path_utm:
                output_data_utm = np.column_stack((slam_data['timestamps'], corrected_pos, corrected_quat))
                np.savetxt(output_path_utm, output_data_utm,
                           fmt=['%.6f'] + ['%.6f'] * 3 + ['%.8f'] * 4, # 时间戳6位小数，位置6位，四元数8位
                           header="timestamp x y z qx qy qz qw (UTM)", comments='')
                print(f"  UTM 坐标轨迹已保存至: {output_path_utm}")

                # --- 尝试保存 WGS84 坐标轨迹 ---
                try:
                    projector = gps_data.get('projector') # 获取之前存储的投影器
                    if projector:
                        print("  正在将修正后的轨迹转换回 WGS84 坐标...")
                        wgs84_lon_lat_alt = utm_to_wgs84(corrected_pos, projector)
                        output_data_wgs84 = np.column_stack((
                            slam_data['timestamps'],
                            wgs84_lon_lat_alt[:, 0], # 经度
                            wgs84_lon_lat_alt[:, 1], # 纬度
                            wgs84_lon_lat_alt[:, 2], # 海拔
                            corrected_quat
                        ))
                        # 生成 WGS84 文件名
                        output_path_wgs84 = output_path_utm.replace('_utm.txt', '_wgs84.txt')
                        if output_path_wgs84 == output_path_utm: # 如果替换失败
                            output_path_wgs84 = output_path_utm.replace('.txt', '_wgs84.txt')

                        np.savetxt(output_path_wgs84, output_data_wgs84,
                                   fmt=['%.6f'] + ['%.8f', '%.8f', '%.3f'] + ['%.8f'] * 4, # 经纬高程和四元数格式
                                   header="timestamp lon lat alt qx qy qz qw (WGS84)", comments='')
                        print(f"  WGS84 坐标轨迹已保存至: {output_path_wgs84}")
                    else:
                        print("警告：GPS 数据中缺少投影仪对象 (projector)，无法保存 WGS84 格式。")
                except Exception as e:
                    print(f"错误：保存 WGS84 格式轨迹失败: {str(e)}")
            else:
                print("  用户取消保存。")

          # 8. 可视化
        print("  正在生成可视化结果图...")
        plot_results(
            original_pos=slam_data['positions'],
            sim3_pos_ekf_input=sim3_pos,          # 传入 Sim3 对齐后的轨迹作为 EKF 输入的参考
            corrected_pos=corrected_pos,          # 传入 EKF 融合后的最终轨迹
            gps_pos=gps_data['positions'],        # 传入过滤后的 GPS 点 (UTM)
            valid_indices_for_error=valid_indices_eval, # 传入用于误差评估的有效索引
            aligned_gps_for_error=eval_gps_points, # 传入用于误差评估的对齐 GPS 点
            slam_times=slam_data['timestamps'],   # 传入 SLAM 时间戳用于时间轴绘图
            ekf_errors=ekf_errors                 # 传入计算出的最终误差
        )

    # --- 异常处理 ---
    except ValueError as ve: # 捕获数据格式、点数不足等错误
        error_msg = f"处理失败 (值错误): {str(ve)}"
        print(f"\n错误!\n{error_msg}")
        if gps_path and not slam_path: error_msg += f"\nGPS 文件: {gps_path}"
        if slam_path and not gps_path: error_msg += f"\nSLAM 文件: {slam_path}"
        if slam_path and gps_path: error_msg += f"\nSLAM 文件: {slam_path}\nGPS 文件: {gps_path}"
        traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except FileNotFoundError as fnf: # 捕获文件未找到错误
        error_msg = f"处理失败 (文件未找到): {str(fnf)}"
        print(f"\n错误!\n{error_msg}"); messagebox.showerror("处理失败", error_msg)
    except AssertionError as ae: # 捕获断言错误 (通常是文件格式不匹配)
        error_msg = f"处理失败 (数据格式断言错误): {str(ae)}"
        print(f"\n错误!\n{error_msg}")
        if gps_path and not slam_path: error_msg += f"\nGPS 文件: {gps_path}"
        if slam_path and not gps_path: error_msg += f"\nSLAM 文件: {slam_path}"
        if slam_path and gps_path: error_msg += f"\nSLAM 文件: {slam_path}\nGPS 文件: {gps_path}"
        traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except CRSError as ce: # 捕获坐标投影错误
         error_msg = f"处理失败 (坐标投影错误): {str(ce)}"
         print(f"\n错误!\n{error_msg}")
         traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except np.linalg.LinAlgError as lae: # 捕获线性代数计算错误 (如 SVD, 求逆失败)
         error_msg = f"处理失败 (线性代数计算错误): {str(lae)}"
         print(f"\n错误!\n{error_msg}")
         traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except RuntimeError as rte: # 捕获由 Sim3 失败等引发的 RuntimeError
         error_msg = f"处理失败 (运行时错误): {str(rte)}"
         print(f"\n错误!\n{error_msg}")
         messagebox.showerror("处理失败", error_msg)
    except Exception as e: # 捕获所有其他未预料的错误
        error_msg = f"处理过程中发生未预料的错误: {type(e).__name__}: {str(e)}"
        print(f"\n严重错误!\n{error_msg}")
        if gps_path and not slam_path: error_msg += f"\nGPS 文件: {gps_path}"
        if slam_path and not gps_path: error_msg += f"\nSLAM 文件: {slam_path}"
        if slam_path and gps_path: error_msg += f"\nSLAM 文件: {slam_path}\nGPS 文件: {gps_path}"
        traceback.print_exc()
        messagebox.showerror("处理失败", f"{error_msg}\n\n详情请查看控制台输出。")

if __name__ == "__main__":
    print("启动 SLAM-GPS 轨迹对齐与融合工具 (EKF+平滑恢复)...")
    print("="*70)
    print("配置参数概览:")
    # 打印 GPS RANSAC 配置
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
    # 打印时间对齐和 Sim3 相关配置
    print(f"  GPS 中断阈值: {CONFIG['time_alignment']['max_gps_gap_threshold']} s")
    print(f"  Sim3 RANSAC 最小内点数: {CONFIG['sim3_ransac']['min_inliers_needed']}")
    print(f"  Sim3 计算使用初始段最大时长: {CONFIG['sim3_ransac']['max_initial_duration']} s") # <<< 打印新配置
    # 打印 EKF 配置
    print(f"  EKF GNSS 恢复平滑步数: {CONFIG['ekf']['transition_steps']}")
    print("="*70)

    main_process_gui()
    print("\n程序结束。")

