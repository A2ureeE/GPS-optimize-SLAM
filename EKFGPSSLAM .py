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
from tkinter import filedialog, messagebox
import traceback
from typing import Dict, Tuple, Optional, List, Any

# ----------------------------
# 配置参数 (修改 EKF 部分，增加 transition_steps)
# ----------------------------
CONFIG = {
    # EKF 参数
    "ekf": {
        "initial_cov_diag": [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01], # x,y,z, qx,qy,qz,qw
        "process_noise_diag": [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01], # Process noise per second (approx) for x,y,z, qx,qy,qz,qw
        "meas_noise_diag": [0.3, 0.3, 0.3], # GPS x,y,z measurement noise std dev
        "transition_steps": 15,         # <<< 新增: GNSS 恢复时，平滑过渡所需的步数
    },
    # Sim(3) 全局变换 RANSAC 参数
    "sim3_ransac": {
        "min_samples": 4,
        "residual_threshold": 0.1,
        "max_trials": 1000,
        "min_inliers_needed": 4,
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
        "max_samples_for_corr": 500,
        "max_gps_gap_threshold": 5.0,
    }
}

# ----------------------------
# Helper Functions (新增/修改)
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
    # Ensure inputs are Rotation objects
    try:
        rot1 = Rotation.from_quat(pose1_quat)
        rot1_inv = rot1.inv()
        rot2 = Rotation.from_quat(pose2_quat)
    except ValueError as e:
        print(f"警告 (calculate_relative_pose): 无效的四元数输入: {e}. 返回零运动。")
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]) # Zero translation, identity rotation

    # 计算在世界坐标系下的位置差
    delta_pos_world = pose2_pos - pose1_pos

    # 将世界坐标系下的位置差转换到 pose1 的局部坐标系下
    delta_pos_local = rot1_inv.apply(delta_pos_world)

    # 计算相对旋转：delta_rot = rot1_inv * rot2
    delta_rot = rot1_inv * rot2
    delta_quat = delta_rot.as_quat() # [x, y, z, w]

    return delta_pos_local, delta_quat

def quaternion_nlerp(q1: np.ndarray, q2: np.ndarray, weight_q2: float) -> np.ndarray:
    """
    对两个四元数进行归一化线性插值 (NLERP)。
    weight_q2 是 q2 的权重 (范围 0 到 1)。
    """
    # 处理方向：确保点积为正，插值走最短路径
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2 # 反转 q2
        dot = -dot

    # 线性插值
    # Clamp weight to [0, 1] for safety
    w = np.clip(weight_q2, 0.0, 1.0)
    q_interp = (1.0 - w) * q1 + w * q2

    # 归一化结果
    norm = np.linalg.norm(q_interp)
    if norm < 1e-9:
        # 如果插值结果接近零（例如权重在边界且输入相同但符号相反），返回其中一个输入
        return q1 if weight_q2 < 0.5 else q2
    return q_interp / norm

# ----------------------------
# 增强数据加载与预处理 (与原版基本一致)
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
            'quaternions': data[:, 4:8].astype(float) # Scipy uses [x, y, z, w]
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
# GPS 离群点过滤模块 (与原版一致)
# -------------------------------------------
def filter_gps_outliers_ransac(times: np.ndarray, positions: np.ndarray,
                               config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 RANSAC 和多项式模型过滤 GPS 轨迹中的离群点。
    支持全局拟合或滑动窗口局部拟合。
    (代码与您提供的版本一致，此处省略以减少篇幅，假设其功能正确)
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

        overall_inlier_mask = np.zeros(n_points, dtype=bool)
        processed_windows = 0
        successful_windows = 0

        start_time = times[0]
        end_time = times[-1]
        current_window_start = start_time

        while current_window_start < end_time:
            current_window_end = current_window_start + window_duration
            window_indices = np.where((times >= current_window_start) & (times < current_window_end))[0]
            n_window_points = len(window_indices)

            if n_window_points >= min_samples_needed:
                processed_windows += 1
                window_times = times[window_indices]
                window_positions = positions[window_indices]
                window_t_feature = window_times.reshape(-1, 1)

                try:
                    window_inlier_masks_dim = []
                    valid_window_fit = True
                    for i in range(positions.shape[1]):
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

                    if valid_window_fit:
                        window_final_inlier_mask = np.logical_and.reduce(window_inlier_masks_dim)
                        original_indices_of_inliers = window_indices[window_final_inlier_mask]
                        overall_inlier_mask[original_indices_of_inliers] = True
                        successful_windows += 1

                except Exception as e:
                     print(f"警告：窗口 [{current_window_start:.2f}s - {current_window_end:.2f}s] RANSAC 拟合失败: {e}")

            if window_step <= 1e-6:
                next_diff_indices = np.where(times > current_window_start)[0]
                if len(next_diff_indices) > 0:
                    current_window_start = times[next_diff_indices[0]]
                else:
                    break
            else:
                 current_window_start += window_step

            if current_window_start >= end_time and times[-1] >= current_window_end :
                 current_window_start = max(start_time, times[-1] - window_duration + 1e-6)

        num_inliers = np.sum(overall_inlier_mask)
        num_outliers = n_points - num_inliers

        print(f"滑动窗口 RANSAC 完成: 处理了 {processed_windows} 个窗口, 其中 {successful_windows} 个成功拟合。")
        if num_outliers > 0:
            print(f"  滑动窗口 RANSAC: 识别并移除了 {num_outliers} 个离群点 (保留 {num_inliers} / {n_points} 个点).")
        else:
            print("  滑动窗口 RANSAC: 未发现离群点。")

        if num_inliers < 2:
             print(f"警告: 滑动窗口 RANSAC 过滤后剩余的 GPS 点数 ({num_inliers}) 过少 (< 2)，可能导致后续处理失败。将返回过滤后的少量点。")
             return times[overall_inlier_mask], positions[overall_inlier_mask]

        return times[overall_inlier_mask], positions[overall_inlier_mask]

# --- load_gps_data, utm_to_wgs84 与原版一致，此处省略 ---
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
# 时间对齐与变换计算 (与原版一致)
# ----------------------------
def estimate_time_offset(slam_times: np.ndarray, gps_times: np.ndarray, max_samples: int) -> float:
    """通过互相关估计时钟偏移 (代码与您提供的版本一致)"""
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
    对每个连续的GPS数据段分别进行三次样条插值，将其插值到对应的SLAM时间戳上
    (代码与您提供的版本一致，此处省略以减少篇幅，假设其功能正确)
    """
    slam_times = slam_data['timestamps']
    gps_times = gps_data['timestamps']     # 过滤后的GPS时间
    gps_positions = gps_data['positions'] # 过滤后的UTM位置
    max_corr_samples = time_align_config['max_samples_for_corr']
    max_gps_gap_threshold = time_align_config['max_gps_gap_threshold'] # 获取中断阈值

    n_slam = len(slam_times)
    n_gps = len(gps_times)

    aligned_gps_full = np.full((n_slam, 3), np.nan)
    valid_mask = np.zeros(n_slam, dtype=bool)

    if n_slam == 0 or n_gps < 2: # GPS至少需要2个点才能插值
        print(f"警告：SLAM 时间戳为空 ({n_slam}) 或 有效GPS点不足 ({n_gps} < 2)，无法进行时间对齐。")
        return aligned_gps_full, valid_mask

    offset = estimate_time_offset(slam_times, gps_times, max_corr_samples)
    adjusted_gps_times = gps_times + offset # 调整GPS时间戳

    try:
        sorted_indices = np.argsort(adjusted_gps_times)
        adjusted_gps_times_sorted = adjusted_gps_times[sorted_indices]
        gps_positions_sorted = gps_positions[sorted_indices]

        unique_times, unique_indices = np.unique(adjusted_gps_times_sorted, return_index=True)
        n_unique_gps = len(unique_times)

        if n_unique_gps < 2:
             print("警告：去重后的有效GPS时间戳少于2个点，无法进行插值。")
             return aligned_gps_full, valid_mask

        if n_unique_gps < n_gps:
            print(f"警告：移除了 {n_gps - n_unique_gps} 个重复的GPS时间戳。")
            adjusted_gps_times_sorted = unique_times
            gps_positions_sorted = gps_positions_sorted[unique_indices]

        time_diffs = np.diff(adjusted_gps_times_sorted)
        gap_indices = np.where(time_diffs > max_gps_gap_threshold)[0] # 找到大于阈值的间隙的 *结束* 索引

        segment_starts = [0] + (gap_indices + 1).tolist() # 每个分段的起始索引 (包括第一个点0)
        segment_ends = gap_indices.tolist() + [n_unique_gps - 1] # 每个分段的结束索引 (包括最后一个点)

        print(f"检测到 {len(gap_indices)} 个 GPS 时间中断 (阈值 > {max_gps_gap_threshold:.1f}s)，将数据分为 {len(segment_starts)} 段进行插值。")

        total_valid_points = 0
        for i in range(len(segment_starts)):
            start_idx = segment_starts[i]
            end_idx = segment_ends[i]
            segment_len = end_idx - start_idx + 1

            if segment_len < 4:
                kind = 'linear' if segment_len >= 2 else None # 点数不足4个用线性，不足2个无法插值
                if kind is None: continue
            else:
                kind = 'cubic'

            segment_times = adjusted_gps_times_sorted[start_idx : end_idx + 1]
            segment_positions = gps_positions_sorted[start_idx : end_idx + 1]
            segment_min_time = segment_times[0]
            segment_max_time = segment_times[-1]

            try:
                if not np.all(np.diff(segment_times) > 0):
                     print(f"警告：段 {i+1} 内时间戳非严格单调递增，跳过此段。")
                     continue

                interp_func_segment = interp1d(
                    segment_times, segment_positions, axis=0, kind=kind,
                    bounds_error=False, fill_value=np.nan
                )
            except ValueError as e:
                print(f"警告：为段 {i+1} 创建插值函数失败 ({kind}插值, {segment_len}点): {e}。跳过此段。")
                continue

            slam_indices_in_segment = np.where(
                (slam_times >= segment_min_time) & (slam_times <= segment_max_time)
            )[0]

            if len(slam_indices_in_segment) > 0:
                interpolated_positions = interp_func_segment(slam_times[slam_indices_in_segment])
                aligned_gps_full[slam_indices_in_segment] = interpolated_positions
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
        return np.full((n_slam, 3), np.nan), np.zeros(n_slam, dtype=bool)

# --- compute_sim3_transform_robust, compute_sim3_transform, transform_trajectory 与原版一致，此处省略 ---
def compute_sim3_transform_robust(src: np.ndarray, dst: np.ndarray,
                                 min_samples: int, residual_threshold: float,
                                 max_trials: int, min_inliers_needed: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """使用 RANSAC 稳健地估计源点(src)到目标点(dst)的 Sim3 变换。"""
    if src.shape[0] < min_samples or dst.shape[0] < min_samples:
        print(f"错误: 点数不足 ({src.shape[0]}) ，无法进行 Sim3 RANSAC (需要至少 {min_samples} 个)")
        return None, None, None
    if src.shape != dst.shape:
         print("错误: Sim3 RANSAC：源点和目标点数量或维度不一致")
         return None, None, None

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    src_norm_factor = np.mean(np.linalg.norm(src_centered, axis=1))
    dst_norm_factor = np.mean(np.linalg.norm(dst_centered, axis=1))

    if src_norm_factor < 1e-9 or dst_norm_factor < 1e-9:
        print("警告：Sim3 RANSAC 输入点集分布非常集中，归一化可能不稳定。跳过归一化。")
        src_normalized = src_centered
        dst_normalized = dst_centered
    else:
        src_normalized = src_centered / src_norm_factor
        dst_normalized = dst_centered / dst_norm_factor

    try:
        ransac_x = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, stop_probability=0.99)
        ransac_y = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, stop_probability=0.99)
        ransac_z = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, stop_probability=0.99)

        ransac_x.fit(src_normalized, dst_normalized[:, 0])
        ransac_y.fit(src_normalized, dst_normalized[:, 1])
        ransac_z.fit(src_normalized, dst_normalized[:, 2])

        inlier_mask = ransac_x.inlier_mask_ & ransac_y.inlier_mask_ & ransac_z.inlier_mask_
        num_inliers = np.sum(inlier_mask)
        print(f"Sim3 RANSAC: 找到 {num_inliers} / {src.shape[0]} 个内点 (阈值={residual_threshold}, 迭代={max_trials})")

        if num_inliers < min_inliers_needed:
            print(f"错误: Sim3 RANSAC: 有效内点不足 ({num_inliers} < {min_inliers_needed})，无法计算可靠的 Sim3 变换")
            return None, None, None

        return compute_sim3_transform(src[inlier_mask], dst[inlier_mask])

    except ValueError as ve:
         print(f"Sim3 RANSAC 失败: {ve}.")
         return None, None, None
    except Exception as e:
         print(f"Sim3 RANSAC 过程中发生未知错误: {e}.")
         traceback.print_exc()
         return None, None, None

def compute_sim3_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """计算从源点(src)到目标点(dst)的最佳 Sim3 变换 (旋转R, 平移t, 尺度s)。"""
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

        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        V = Vt.T
        R = V @ U.T

        if np.linalg.det(R) < 0:
            print("警告: 检测到反射（行列式为负），修正旋转矩阵。")
            Vt_copy = Vt.copy()
            Vt_copy[-1, :] *= -1
            R = Vt_copy.T @ U.T

        src_dist_sq = np.sum(src_centered**2, axis=1)
        numerator = np.sum(np.sum(dst_centered * (src_centered @ R.T), axis=1))
        denominator = np.sum(src_dist_sq)

        if denominator < 1e-9:
             print("警告：源点非常集中于质心，无法可靠计算尺度，默认为 1.0")
             scale = 1.0
        else:
            scale = numerator / denominator
            if scale <= 1e-6:
                 print(f"警告：计算出的尺度非常小 ({scale:.2e})，可能存在问题。重置为 1.0")
                 scale = 1.0

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

def transform_trajectory(positions: np.ndarray, quaternions: np.ndarray,
                        R: np.ndarray, t: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """应用Sim3变换到整个轨迹（位置和姿态）"""
    trans_pos = scale * (positions @ R.T) + t

    R_sim3_rot = Rotation.from_matrix(R)
    trans_quat_list = []
    for q_xyzw in quaternions:
        original_rot = Rotation.from_quat(q_xyzw)
        new_rot = R_sim3_rot * original_rot
        new_quat_xyzw = new_rot.as_quat()
        trans_quat_list.append(new_quat_xyzw)

    return trans_pos, np.array(trans_quat_list)

# --- 可视化与评估系统 plot_results 与原版一致，此处省略 ---
def plot_results(original_pos: np.ndarray, corrected_pos: np.ndarray,
                 gps_pos: np.ndarray, # 原始(过滤后)的GPS UTM点
                 valid_indices_for_error: np.ndarray,
                 aligned_gps_for_error: np.ndarray, # 对齐到SLAM时间的GPS点
                 slam_times: np.ndarray, # SLAM 时间戳用于绘图
                 ekf_errors: Optional[np.ndarray] = None # EKF 最终误差 (可能为空)
                ) -> None:
    """增强的可视化系统，显示原始轨迹、修正后轨迹、GPS点和误差"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告：未能设置中文字体 'SimHei'。标签可能显示为方块。请确保已安装该字体。")

    plt.figure(figsize=(15, 12))
    plt.suptitle("SLAM-GPS 轨迹对齐与融合结果", fontsize=16)

    # --- 1. 2D轨迹对比 (XY平面) ---
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(original_pos[:, 0], original_pos[:, 1], 'b--', alpha=0.6, linewidth=1, label='原始 SLAM 轨迹')
    ax1.plot(corrected_pos[:, 0], corrected_pos[:, 1], 'g-', linewidth=1.5, label='EKF 修正后轨迹')
    ax1.scatter(gps_pos[:, 0], gps_pos[:, 1], c='r', marker='x', s=30, label='GPS 参考点 (过滤后, UTM)')
    step = max(1, len(valid_indices_for_error) // 100)
    if len(valid_indices_for_error) > 0:
       ax1.scatter(aligned_gps_for_error[::step, 0], aligned_gps_for_error[::step, 1],
                   facecolors='none', edgecolors='orange', marker='o', s=40,
                   label='对齐GPS点 (插值, 用于误差评估)')

    ax1.set_title('轨迹对比 (XY平面)')
    ax1.set_xlabel('X (米)')
    ax1.set_ylabel('Y (米)')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

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
    try:
        max_range = np.array([corrected_pos[:,0].max()-corrected_pos[:,0].min(),
                              corrected_pos[:,1].max()-corrected_pos[:,1].min(),
                              corrected_pos[:,2].max()-corrected_pos[:,2].min()]).max() / 2.0 * 1.1
        if max_range < 1.0: max_range = 5.0
        mid_x = np.median(corrected_pos[:,0])
        mid_y = np.median(corrected_pos[:,1])
        mid_z = np.median(corrected_pos[:,2])
        ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    except:
        pass

    # --- 3. 最终误差分布 (直方图) ---
    ax_hist = plt.subplot(2, 2, 3)
    if ekf_errors is not None and len(ekf_errors) > 0:
        mean_err = np.mean(ekf_errors)
        std_err = np.std(ekf_errors)
        max_err = np.max(ekf_errors)
        median_err = np.median(ekf_errors)

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


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- GUI文件选择功能 与原版一致，此处省略 ---
def select_file_dialog(title: str, filetypes: List[Tuple[str, str]]) -> str:
    """通用文件选择对话框"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
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
# EKF 实现 (修改后)
# ----------------------------

class ExtendedKalmanFilter:
    def __init__(self, initial_pos: np.ndarray, initial_quat: np.ndarray,
                 config: Dict[str, Any]): # Pass full EKF config
        """
        初始化扩展卡尔曼滤波器 (实现航位推算+平滑恢复逻辑)
        状态向量 (state): [x, y, z, qx, qy, qz, qw] (7维)
        """
        if not (initial_pos.shape == (3,) and initial_quat.shape == (4,)):
            raise ValueError(f"EKF 初始化: 初始位置或四元数维度错误")

        ekf_config = config # Use the passed config directly

        initial_quat_normalized = self.normalize_quaternion(initial_quat)
        self.state = np.concatenate([initial_pos, initial_quat_normalized]).astype(float)
        self.cov = np.diag(ekf_config['initial_cov_diag']).astype(float)
        self.Q_per_sec = np.diag(ekf_config['process_noise_diag']).astype(float) # Process noise per second
        self.R = np.diag(ekf_config['meas_noise_diag']).astype(float)

        if self.state.shape != (7,): raise ValueError(f"EKF: 内部状态维度错误")
        if self.cov.shape != (7, 7): raise ValueError(f"EKF: 内部协方差维度错误")
        if self.Q_per_sec.shape != (7, 7): raise ValueError(f"EKF: 内部过程噪声Q维度错误")
        if self.R.shape != (3, 3): raise ValueError(f"EKF: 内部测量噪声R维度错误")

        # --- 新增: 平滑状态 ---
        self.gnss_available_prev = None
        self.gnss_update_weight = 0.0 # GNSS 更新贡献的权重 (0.0 to 1.0)
        self.transition_steps = max(1, int(ekf_config.get('transition_steps', 10))) # 从配置读取
        self.weight_delta = 1.0 / self.transition_steps if self.transition_steps > 0 else 1.0
        self._last_predicted_state = self.state.copy() # 存储预测结果用于平滑

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
            slam_motion_update: (delta_pos_local, delta_quat) from calculate_relative_pose
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
        # 姿态预测: q_pred = q_prev * delta_q
        predicted_rot = prev_rot * delta_rot
        predicted_quat = predicted_rot.as_quat() # [x, y, z, w]

        predicted_state = np.concatenate([predicted_pos, predicted_quat])

        # --- 3. 计算状态转移矩阵 Jacobian F (7x7) ---
        # F = ∂f/∂x | evaluated at prev_state, motion_update
        # This is complex for 7D state. We approximate it here.
        # For position part: ∂pos_pred/∂pos_prev = I (3x3)
        #                  ∂pos_pred/∂quat_prev depends on derivative of rotation matrix wrt quaternion (complex)
        # For quat part: ∂quat_pred/∂pos_prev = 0 (3x4)
        #                ∂quat_pred/∂quat_prev involves derivative of quaternion multiplication (complex)
        # Simplified F (assuming orientation error effect on position prediction is in Q):
        F = np.eye(7)
        # Add term for rotation effect on position prediction (more accurate)
        # Need skew-symmetric matrix of R(q_prev) @ delta_pos_local
        # sk = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) where [x,y,z] = R(q_prev) @ delta_pos_local
        rotated_delta_pos = prev_rot.apply(delta_pos_local)
        x, y, z = rotated_delta_pos
        skew_rotated_delta_pos = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        # Approximate dpos/dq = -skew(R@delta_p) * dR/dq * dq (dR/dq is complex)
        # Often simplified or effects absorbed into Q. Let's use simplified F=eye(7) for now.

        # --- 4. 调整过程噪声 Q for delta_time ---
        dt_adjusted = max(abs(delta_time), 1e-6)
        Q = self.Q_per_sec * dt_adjusted # Scale Q by time interval

        # --- 5. 预测协方差 ---
        # P_pred = F * P_prev * F^T + Q
        # Using F = eye(7) simplifies this to P_pred = P_prev + Q
        predicted_covariance = prev_cov + Q
        predicted_covariance = (predicted_covariance + predicted_covariance.T) / 2.0 # Ensure symmetry

        # 缓存预测结果
        self._last_predicted_state = predicted_state.copy()

        return predicted_state, predicted_covariance

    def _update(self, predicted_state: np.ndarray, predicted_covariance: np.ndarray,
                gps_pos: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        EKF 更新步骤：使用 GPS 位置测量修正状态估计。
        Args:
            predicted_state, predicted_covariance: from _predict step
            gps_pos: GPS measurement [x, y, z]
        Returns:
            (updated_state, updated_covariance) or (None, None) if update fails
        """
        if gps_pos.shape != (3,) or np.isnan(gps_pos).any():
            return None, None # Invalid measurement

        # 测量矩阵 H (观测模型)：只观测位置 x, y, z
        H = np.zeros((3, 7))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        try:
            # 1. 计算测量残差 (Innovation): y = z_gps - H * x_predicted
            innovation = gps_pos - predicted_state[:3] # H * x_predicted is just the position part

            # 2. 计算残差协方差 (Innovation Covariance): S = H * P_pred * H^T + R
            H_P = H @ predicted_covariance
            S = H_P @ H.T + self.R
            S = (S + S.T) / 2.0

            # 检查 S 是否奇异
            if abs(np.linalg.det(S)) < 1e-12:
                # print(f"警告 (EKF Update): S 矩阵接近奇异，跳过更新。")
                return None, None

            # 3. 计算卡尔曼增益 (Kalman Gain): K = P_pred * H^T * S^(-1)
            S_inv = np.linalg.inv(S)
            K = predicted_covariance @ H.T @ S_inv

            # 4. 状态更新: x_new = x_predicted + K * y
            updated_state = predicted_state + K @ innovation
            # 更新后必须重新归一化四元数部分
            updated_state[3:] = self.normalize_quaternion(updated_state[3:])

            # 5. 协方差更新: P_new = (I - K * H) * P_predicted (or Joseph form)
            I = np.eye(7)
            # Joseph form (更稳定)
            P_new = (I - K @ H) @ predicted_covariance @ (I - K @ H).T + K @ self.R @ K.T
            # 标准形式: P_new = (I - K @ H) @ predicted_covariance
            updated_covariance = (P_new + P_new.T) / 2.0

            return updated_state, updated_covariance

        except np.linalg.LinAlgError as e:
            print(f"警告 (EKF Update): 更新步骤中发生线性代数错误: {e}。跳过更新。")
            return None, None
        except Exception as e:
            print(f"警告 (EKF Update): 更新步骤中发生未知错误: {e}。跳过更新。")
            traceback.print_exc()
            return None, None

    def process_step(self, slam_motion_update: Tuple[np.ndarray, np.ndarray],
                     gps_measurement: Optional[np.ndarray], gnss_available: bool,
                     delta_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理一个时间步：预测、(可能)更新、(可能)平滑，并更新内部状态。
        Returns:
            final_fused_pose (pos, quat): 当前步骤的最佳估计位姿。
        """
        # --- 1. EKF 预测 ---
        predicted_state, predicted_covariance = self._predict(slam_motion_update, delta_time)

        # --- 2. EKF 更新 (如果GNSS可用) ---
        updated_state = None
        updated_covariance = None
        if gnss_available and gps_measurement is not None:
            res = self._update(predicted_state, predicted_covariance, gps_measurement)
            if res[0] is not None: # Check if update was successful
                updated_state, updated_covariance = res

        # --- 3. 处理GNSS状态变化和平滑权重 ---
        just_recovered = gnss_available and (self.gnss_available_prev == False)

        if gnss_available:
            if just_recovered:
                self.gnss_update_weight = self.weight_delta
            elif self.gnss_update_weight < 1.0:
                self.gnss_update_weight = min(1.0, self.gnss_update_weight + self.weight_delta)
            else:
                self.gnss_update_weight = 1.0
        else: # GNSS 不可用
            self.gnss_update_weight = 0.0

        # --- 4. 计算最终融合状态 (应用平滑) ---
        final_fused_state = predicted_state
        final_fused_covariance = predicted_covariance

        # 平滑仅在 GNSS 可用、更新成功且权重<1 时进行
        if gnss_available and updated_state is not None and self.gnss_update_weight < 1.0:
            w = self.gnss_update_weight # 更新结果的权重
            pred_state_for_smooth = self._last_predicted_state # 使用缓存的预测结果

            # 位置插值
            smooth_pos = (1.0 - w) * pred_state_for_smooth[:3] + w * updated_state[:3]
            # 姿态插值 (NLERP)
            smooth_quat = quaternion_nlerp(pred_state_for_smooth[3:], updated_state[3:], w)

            final_fused_state = np.concatenate([smooth_pos, smooth_quat])
            # 平滑期间的协方差: 简单地使用更新后的协方差 (更优估计)
            final_fused_covariance = updated_covariance

        # 如果 GNSS 可用且平滑完成 (w=1) 或刚更新，则使用更新结果
        elif gnss_available and updated_state is not None:
             final_fused_state = updated_state
             final_fused_covariance = updated_covariance

        # 如果 GNSS 不可用，则使用预测结果 (已在初始化时设置)

        # --- 5. 更新EKF内部状态 ---
        self.state = final_fused_state.copy()
        self.cov = final_fused_covariance.copy()

        # --- 6. 更新上一步GNSS状态 ---
        self.gnss_available_prev = gnss_available

        # --- 7. 返回当前步最终融合结果 ---
        return self.state[:3].copy(), self.state[3:].copy() # pos, quat

# ----------------------------
# EKF 轨迹修正应用函数 (修改后)
# ----------------------------

def apply_ekf_correction(slam_data: Dict[str, np.ndarray], gps_data: Dict[str, np.ndarray],
                         sim3_pos: np.ndarray, sim3_quat: np.ndarray,
                         config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用 EKF 对经过 Sim3 全局变换的轨迹进行逐点融合修正。
    现在使用内部包含平滑逻辑的 EKF 类。
    """
    n_points = len(slam_data['timestamps'])
    if n_points == 0:
        print("错误 (apply_ekf_correction): 输入的 SLAM 数据为空。")
        return np.empty((0, 3)), np.empty((0, 4))
    if not (sim3_pos.shape[0] == n_points and sim3_quat.shape[0] == n_points):
         raise ValueError(f"错误: Sim3 变换后的轨迹点数与原始 SLAM 时间戳数量不匹配。")

    ekf_config = config['ekf'] # 获取 EKF 配置
    time_align_config = config['time_alignment'] # 获取时间对齐配置

    # 1. 初始化 EKF (传入 EKF 相关配置)
    try:
        ekf = ExtendedKalmanFilter(
            initial_pos=sim3_pos[0],
            initial_quat=sim3_quat[0],
            config=ekf_config # Pass the EKF sub-dictionary
        )
        # 设置 EKF 的初始 gnss_available_prev 状态 (基于第一个点是否有GPS)
        aligned_gps_for_init, valid_mask_for_init = dynamic_time_alignment(
            slam_data, gps_data, time_align_config
        )
        ekf.gnss_available_prev = valid_mask_for_init[0] if n_points > 0 else False
        print(f"EKF 初始化完成。初始 GNSS 状态: {ekf.gnss_available_prev}")

    except Exception as e:
         raise ValueError(f"EKF 初始化失败: {e}")

    # 2. 获取对齐的 GPS 数据用于 EKF 更新 (与之前相同)
    print("正在执行分段时间对齐以获取 EKF 更新所需的 GPS 测量...")
    # 使用与初始化时相同的对齐结果
    aligned_gps_for_update = aligned_gps_for_init
    valid_mask_for_update = valid_mask_for_init
    num_valid_gps_for_update = np.sum(valid_mask_for_update)
    print(f"EKF 修正：共有 {num_valid_gps_for_update} / {n_points} 个时间点将尝试使用有效的 GPS 测量进行更新。")

    # 准备存储修正后的轨迹
    corrected_pos_list = np.zeros_like(sim3_pos)
    corrected_quat_list = np.zeros_like(sim3_quat)

    # 第一个点的状态就是初始状态 (已在 EKF 内部设置)
    corrected_pos_list[0], corrected_quat_list[0] = ekf.state[:3].copy(), ekf.state[3:].copy()

    # --- 获取原始 SLAM 位姿用于计算相对运动 ---
    orig_slam_pos = slam_data['positions']
    orig_slam_quat = slam_data['quaternions']

    last_time = slam_data['timestamps'][0]
    predict_steps = 0
    updates_attempted = 0 # 尝试更新次数 (有GPS测量)

    # 3. 迭代处理每个 SLAM 时间点 (从第二个点开始)
    for i in range(1, n_points):
        current_time = slam_data['timestamps'][i]
        delta_t = current_time - last_time
        if delta_t <= 1e-9: # 避免零或负 dt
            # print(f"警告 (EKF loop): 时间戳间隔过小或非单调在索引 {i} (dt={delta_t:.4f}s)。使用小正值。")
            delta_t = 1e-6 # 使用一个非常小的正时间间隔，Q 影响会很小

        # a) 计算 SLAM 的相对运动 (Motion Update)
        # 使用 *原始* SLAM 数据计算相对运动
        slam_motion_update = calculate_relative_pose(
            orig_slam_pos[i-1], orig_slam_quat[i-1],
            orig_slam_pos[i], orig_slam_quat[i]
        )

        # b) 获取当前时间点的 GPS 测量和可用性
        gnss_available = valid_mask_for_update[i]
        gps_measurement = aligned_gps_for_update[i] if gnss_available else None
        if gps_measurement is not None and np.isnan(gps_measurement).any():
            # 如果插值结果意外为 NaN，视为无效
            gnss_available = False
            gps_measurement = None

        # c) 调用 EKF 的处理步骤
        # process_step 内部会处理预测、更新、平滑，并更新 EKF 的内部状态
        fused_pos, fused_quat = ekf.process_step(
            slam_motion_update=slam_motion_update,
            gps_measurement=gps_measurement,
            gnss_available=gnss_available,
            delta_time=delta_t
        )
        predict_steps += 1
        if gnss_available:
             updates_attempted += 1

        # d) 记录当前 EKF 修正后的状态
        corrected_pos_list[i] = fused_pos
        corrected_quat_list[i] = fused_quat
        last_time = current_time

    print(f"EKF 修正完成。共执行 {predict_steps} 步处理。")
    print(f"  尝试进行 GPS 更新的步数: {updates_attempted} / {predict_steps}")
    # (内部的成功/失败/跳过次数可以在 EKF 类中添加计数器来获取更详细信息)

    return corrected_pos_list, corrected_quat_list

# ----------------------------
# 主流程控制 (与原版基本一致，调用修改后的函数)
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
        valid_indices_sim3 = np.where(valid_mask_sim3)[0]
        min_points_needed_for_sim3 = CONFIG['sim3_ransac']['min_inliers_needed']
        if len(valid_indices_sim3) < min_points_needed_for_sim3:
            raise ValueError(f"有效时间同步匹配点不足 ({len(valid_indices_sim3)} < {min_points_needed_for_sim3})，无法进行 Sim3 变换估计。")
        print(f"找到 {len(valid_indices_sim3)} 个有效时间同步的匹配点用于 Sim3 估计。")
        print("时间对齐完成。")
        print("-" * 30)

        # 4. Sim3 全局对齐
        print("步骤 3/7: 计算稳健的 Sim3 全局变换...")
        src_points = slam_data['positions'][valid_indices_sim3]
        dst_points = aligned_gps_for_sim3[valid_indices_sim3]
        R, t, scale = compute_sim3_transform_robust(
            src=src_points, dst=dst_points, **CONFIG['sim3_ransac'] # 使用字典解包传递参数
        )
        if R is None or t is None or scale is None:
             raise RuntimeError("Sim3 全局变换计算失败，无法继续。")
        print("Sim3 全局变换计算成功。")
        print("步骤 4/7: 应用 Sim3 变换到整个 SLAM 轨迹...")
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
        aligned_gps_for_eval, valid_mask_eval = dynamic_time_alignment(
             slam_data, gps_data, CONFIG['time_alignment']
        )
        valid_indices_eval = np.where(valid_mask_eval)[0]
        eval_gps_points = np.empty((0,3))
        ekf_errors = None
        # ... (误差计算部分与原版一致) ...
        if len(valid_indices_eval) > 0:
            eval_gps_points = aligned_gps_for_eval[valid_indices_eval]
            print(f"  (基于 {len(valid_indices_eval)} 个有效对齐点进行评估)")
            raw_positions_eval = slam_data['positions'][valid_indices_eval]
            raw_errors = np.linalg.norm(raw_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] 原始轨迹   vs 对齐GPS -> 均值误差: {np.mean(raw_errors):.3f} m, 中位数误差: {np.median(raw_errors):.3f} m")
            sim3_positions_eval = sim3_pos[valid_indices_eval]
            sim3_errors = np.linalg.norm(sim3_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] Sim3 对齐后 vs 对齐GPS -> 均值误差: {np.mean(sim3_errors):.3f} m, 中位数误差: {np.median(sim3_errors):.3f} m")
            ekf_positions_eval = corrected_pos[valid_indices_eval]
            ekf_errors = np.linalg.norm(ekf_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] EKF 融合后  vs 对齐GPS -> 均值误差: {np.mean(ekf_errors):.3f} m, 中位数误差: {np.median(ekf_errors):.3f} m")
        else:
            print("警告：无有效对齐的GPS点用于最终误差评估。")
            ekf_errors = None
        print("误差评估完成。")
        print("-" * 30)

        # 7. 保存结果
        print("步骤 7/7: 保存结果与可视化...")
        save_results = messagebox.askyesno("保存结果", "处理完成。是否要保存修正后的轨迹?")
        if save_results:
            base_filename = slam_path.split('/')[-1].split('\\')[-1]
            default_save_name = base_filename.replace('.txt', '_corrected_utm.txt')
            if default_save_name == base_filename: default_save_name += '_corrected_utm.txt'

            output_path_utm = filedialog.asksaveasfilename(
                title="保存修正后的轨迹 (UTM 坐标, TUM 格式)", defaultextension=".txt",
                filetypes=[("文本文件", "*.txt")], initialfile=default_save_name
            )
            if output_path_utm:
                output_data_utm = np.column_stack((slam_data['timestamps'], corrected_pos, corrected_quat))
                np.savetxt(output_path_utm, output_data_utm,
                           fmt=['%.6f'] + ['%.6f'] * 3 + ['%.8f'] * 4,
                           header="timestamp x y z qx qy qz qw (UTM)", comments='')
                print(f"  UTM 坐标轨迹已保存至: {output_path_utm}")
                # --- 尝试保存 WGS84 ---
                try:
                    projector = gps_data.get('projector')
                    if projector:
                        print("  正在将修正后的轨迹转换回 WGS84 坐标...")
                        wgs84_lon_lat_alt = utm_to_wgs84(corrected_pos, projector)
                        output_data_wgs84 = np.column_stack((
                            slam_data['timestamps'], wgs84_lon_lat_alt[:, 0], wgs84_lon_lat_alt[:, 1], wgs84_lon_lat_alt[:, 2], corrected_quat
                        ))
                        output_path_wgs84 = output_path_utm.replace('_utm.txt', '_wgs84.txt')
                        if output_path_wgs84 == output_path_utm: output_path_wgs84 = output_path_utm.replace('.txt', '_wgs84.txt')
                        np.savetxt(output_path_wgs84, output_data_wgs84,
                                   fmt=['%.6f'] + ['%.8f', '%.8f', '%.3f'] + ['%.8f'] * 4,
                                   header="timestamp lon lat alt qx qy qz qw (WGS84)", comments='')
                        print(f"  WGS84 坐标轨迹已保存至: {output_path_wgs84}")
                    else: print("警告：GPS 数据中缺少投影仪对象，无法保存 WGS84 格式。")
                except Exception as e: print(f"错误：保存 WGS84 格式轨迹失败: {str(e)}")
            else: print("  用户取消保存。")

        # 8. 可视化
        print("  正在生成可视化结果图...")
        plot_results(
            original_pos=slam_data['positions'], corrected_pos=corrected_pos,
            gps_pos=gps_data['positions'], valid_indices_for_error=valid_indices_eval,
            aligned_gps_for_error=eval_gps_points, slam_times=slam_data['timestamps'],
            ekf_errors=ekf_errors
        )

        print("-" * 30)
        print("所有处理步骤完成。")
        messagebox.showinfo("完成", "轨迹对齐与融合处理完成！")

    # --- Exception Handling (与原版一致) ---
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
    print("启动 SLAM-GPS 轨迹对齐与融合工具 (实现DR+平滑恢复)...") # 修改启动信息
    print("="*70)
    print("配置参数概览:")
    # ... (打印 GPS RANSAC 配置) ...
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
    print(f"  Sim3 RANSAC 最小内点数: {CONFIG['sim3_ransac']['min_inliers_needed']}")
    print(f"  EKF GNSS 恢复平滑步数: {CONFIG['ekf']['transition_steps']}") # 打印新参数
    print("="*70)

    main_process_gui()
    print("\n程序结束。")
