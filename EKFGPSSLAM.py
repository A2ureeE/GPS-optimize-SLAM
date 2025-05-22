# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from scipy.spatial import distance 
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
        "process_noise_diag": [0.1, 0.1, 0.7, 0.01, 0.01, 0.01, 0.01], # 每秒过程噪声 x,y,z, qx,qy,qz,qw
        "meas_noise_diag": [0.2, 0.2, 0.2], # GPS x,y,z 测量噪声标准差
        "transition_steps": 10, # GNSS恢复时的平滑过渡步数 (当不使用RTS时)
                               # 注意：如果决定使用RTS，此值在恢复点逻辑上应为0
    },
    # Sim(3) 全局变换 RANSAC 参数
    "sim3_ransac": {
        "min_samples": 4,               # RANSAC 模型最小样本数
        "residual_threshold": 4.0,      # RANSAC 内点阈值 (米)
        "max_trials": 1000,             # RANSAC 最大迭代次数
        "min_inliers_needed": 4,        # 有效Sim3拟合所需最小内点数
        "max_initial_duration": 180.0,   # 用于Sim3计算的初始段最大时长 (秒)
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
        "max_samples_for_corr": 500,    # 初始时间偏移互相关的最大样本数
        "max_gps_gap_threshold": 5.0,   # 定义GPS段断裂的最大时间间隙 (秒)
    },
    # 地面真值GPS滤波参数
    "ground_truth_gps_filtering": { 
        "enabled": False, 
        "use_sliding_window": True,
        "window_duration_seconds": 15.0,
        "window_step_factor": 0.5,
        "polynomial_degree": 2,
        "min_samples": 6,
        "residual_threshold_meters": 5.0,
        "max_trials": 50,
    },
    # RTS决策参数
    "rts_decision": {
        "sharp_turn_yaw_rate_threshold_deg_per_sec": 45.0, # 每秒偏航角变化超过此值视为急转弯
        "default_ekf_transition_steps_on_sharp_turn": 0 # 急转弯时，EKF采用的过渡步数
    }
}

# ----------------------------
# 辅助函数
# ----------------------------

def calculate_relative_pose(pose1_pos: np.ndarray, pose1_quat: np.ndarray,
                            pose2_pos: np.ndarray, pose2_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算从位姿1到位姿2的相对运动。"""
    try:
        rot1 = Rotation.from_quat(pose1_quat)
        rot1_inv = rot1.inv()
        rot2 = Rotation.from_quat(pose2_quat)
    except ValueError as e:
        print(f"警告 (calculate_relative_pose): 无效的四元数输入: {e}。返回零运动。")
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])

    delta_pos_world = pose2_pos - pose1_pos
    delta_pos_local = rot1_inv.apply(delta_pos_world)
    delta_rot = rot1_inv * rot2
    delta_quat = delta_rot.as_quat()
    return delta_pos_local, delta_quat

def quaternion_nlerp(q1: np.ndarray, q2: np.ndarray, weight_q2: float) -> np.ndarray:
    """在两个四元数之间执行归一化线性插值 (NLERP)。"""
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        # dot = -dot # Not needed after flipping q2
    w = np.clip(weight_q2, 0.0, 1.0)
    q_interp = (1.0 - w) * q1 + w * q2
    norm = np.linalg.norm(q_interp)
    if norm < 1e-9:
        return q1 if weight_q2 < 0.5 else q2
    return q_interp / norm

# ----------------------------
# 数据加载和预处理
# ----------------------------
def load_slam_trajectory(txt_path: str) -> Dict[str, np.ndarray]:
    """加载并验证SLAM轨迹数据 (TUM格式)。"""
    try:
        data = np.loadtxt(txt_path)
        if data.ndim == 1: data = data.reshape(1, -1)
        if data.shape[1] != 8:
             raise ValueError(f"SLAM文件格式错误: 应为8列 (ts x y z qx qy qz qw)，实际为 {data.shape[1]} 列")
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
    """根据经度自动计算UTM区域号和半球。"""
    if lons.size == 0 or lats.size == 0:
        raise ValueError("经纬度数据不能为空以确定UTM区域。")
    central_lon = np.mean(lons)
    zone = int((central_lon + 180) // 6 + 1)
    hemisphere = ' +south' if np.mean(lats) < 0 else ''
    return zone, hemisphere

def filter_gps_outliers_ransac(times: np.ndarray, positions: np.ndarray,
                               config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """使用RANSAC和多项式模型过滤GPS轨迹中的异常值。"""
    if not config.get("enabled", False):
        print("此数据集的GPS RANSAC滤波已禁用。")
        return times, positions
    n_points = len(times)
    min_samples_needed = config['min_samples']
    if n_points < min_samples_needed:
        print(f"警告: GPS点数 ({n_points}) 少于RANSAC最小样本数 ({min_samples_needed})。跳过GPS异常值滤波。")
        return times, positions

    use_sliding_window = config.get("use_sliding_window", False)
    if not use_sliding_window:
        # ... (全局RANSAC逻辑，保持不变，输出中文)
        print("执行全局GPS RANSAC滤波...")
        try:
            t_feature = times.reshape(-1, 1)
            inlier_masks = []
            for i in range(positions.shape[1]): # 分别拟合X, Y, Z
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

            final_inlier_mask = np.logical_and.reduce(inlier_masks)
            num_inliers = np.sum(final_inlier_mask)
            num_outliers = n_points - num_inliers

            if num_outliers > 0:
                print(f"  全局RANSAC: 识别并移除了 {num_outliers} 个异常点 (保留 {num_inliers}/{n_points} 个点)。")
            else:
                print("  全局RANSAC: 未找到异常点。")
            if num_inliers < min_samples_needed:
                 print(f"警告: 全局RANSAC滤波后剩余GPS点数 ({num_inliers}) 过少。")
            return times[final_inlier_mask], positions[final_inlier_mask]
        except Exception as e:
            print(f"全局GPS RANSAC滤波过程中出错: {e}。跳过滤波。")
            traceback.print_exc()
            return times, positions
    else: # 滑动窗口 RANSAC
        # ... (滑动窗口RANSAC逻辑，保持不变，输出中文)
        window_duration = config['window_duration_seconds']
        step_factor = config['window_step_factor']
        window_step = window_duration * step_factor
        residual_threshold = config['residual_threshold_meters']
        poly_degree = config['polynomial_degree']
        max_trials_per_window = config['max_trials']
        print(f"执行滑动窗口GPS RANSAC滤波 (窗口: {window_duration}s, 步长: {window_step:.2f}s)...")
        if n_points < min_samples_needed:
             print(f"警告: 总点数 ({n_points}) 少于窗口最小样本数 ({min_samples_needed})。无法使用滑动窗口。")
             return times, positions
        overall_inlier_mask = np.zeros(n_points, dtype=bool)
        processed_windows = 0
        successful_windows = 0
        start_time_data = times[0]
        end_time_data = times[-1]
        current_window_start = start_time_data
        while current_window_start < end_time_data:
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
                    for i_dim in range(positions.shape[1]):
                        target = window_positions[:, i_dim]
                        model = make_pipeline(
                            PolynomialFeatures(degree=poly_degree),
                            RANSACRegressor(
                                min_samples=min_samples_needed,
                                residual_threshold=residual_threshold,
                                max_trials=max_trials_per_window,
                            )
                        )
                        model.fit(window_t_feature, target)
                        window_inlier_masks_dim.append(model[-1].inlier_mask_)
                    window_final_inlier_mask = np.logical_and.reduce(window_inlier_masks_dim)
                    original_indices_of_inliers = window_indices[window_final_inlier_mask]
                    overall_inlier_mask[original_indices_of_inliers] = True
                    successful_windows += 1
                except Exception as e_win:
                     print(f"警告: 窗口 [{current_window_start:.2f}s - {current_window_end:.2f}s] RANSAC拟合失败: {e_win}")
            if window_step <= 1e-6: 
                next_diff_indices = np.where(times > current_window_start)[0]
                if len(next_diff_indices) > 0: current_window_start = times[next_diff_indices[0]]
                else: break 
            else: current_window_start += window_step
            if current_window_start >= end_time_data and times[-1] >= current_window_end : 
                 current_window_start = max(start_time_data, times[-1] - window_duration + 1e-6) 
        num_inliers = np.sum(overall_inlier_mask)
        num_outliers = n_points - num_inliers
        print(f"滑动窗口RANSAC完成: 处理了 {processed_windows} 个窗口, {successful_windows} 次成功拟合。")
        if num_outliers > 0:
            print(f"  滑动窗口RANSAC: 移除了 {num_outliers} 个异常点 (保留 {num_inliers}/{n_points} 个点)。")
        else:
            print("  滑动窗口RANSAC: 未找到异常点。")
        if num_inliers < 2: 
             print(f"警告: 滑动窗口RANSAC后剩余GPS点数 ({num_inliers}) 过少。")
        return times[overall_inlier_mask], positions[overall_inlier_mask]

def load_gps_data(txt_path: str, data_label: str = "GPS", filter_config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """加载GPS数据，执行UTM投影，并可选地过滤异常值。"""
    try:
        try: gps_data_raw = np.loadtxt(txt_path, delimiter=' ')
        except ValueError: gps_data_raw = np.loadtxt(txt_path, delimiter=',')
        if gps_data_raw.ndim == 1: gps_data_raw = gps_data_raw.reshape(1, -1)
        if gps_data_raw.shape[1] < 4: 
             raise ValueError(f"{data_label} 文件至少需要4列 (ts lat lon alt)，实际为 {gps_data_raw.shape[1]} 列")

        timestamps_raw, lats_raw, lons_raw, alts_raw = gps_data_raw[:,0], gps_data_raw[:,1], gps_data_raw[:,2], gps_data_raw[:,3]
        valid_gps_mask = (np.abs(lats_raw) <= 90) & (np.abs(lons_raw) <= 180) & (lats_raw != 0) & (lons_raw != 0)
        if not np.all(valid_gps_mask):
            num_filtered_invalid = len(lats_raw) - np.sum(valid_gps_mask)
            print(f"警告 ({data_label}): 过滤了 {num_filtered_invalid} 个无效的GPS经纬度点 (超出范围或为零)。")
            timestamps_raw, lats_raw, lons_raw, alts_raw = timestamps_raw[valid_gps_mask], lats_raw[valid_gps_mask], lons_raw[valid_gps_mask], alts_raw[valid_gps_mask]
            if len(timestamps_raw) == 0: raise ValueError(f"{data_label}: 过滤无效经纬度后无有效GPS数据。")

        utm_zone_number, utm_hemisphere = auto_utm_projection(lons_raw, lats_raw)
        proj_string = f"+proj=utm +zone={utm_zone_number}{utm_hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        try: projector = Proj(proj_string)
        except CRSError as e: raise ValueError(f"无法为 {data_label} 创建UTM投影 (区域: {utm_zone_number}{utm_hemisphere})。Proj错误: {e}")
        x, y = projector(lons_raw, lats_raw)
        utm_positions_raw = np.column_stack((x, y, alts_raw))

        current_filter_config = filter_config_override if filter_config_override is not None else CONFIG['gps_filtering_ransac']
        print(f"为 {data_label} 轨迹执行RANSAC滤波...")
        filtered_times, filtered_utm_positions = filter_gps_outliers_ransac(timestamps_raw, utm_positions_raw, current_filter_config)
        
        ransac_removed_count = len(timestamps_raw) - len(filtered_times)
        if ransac_removed_count > 0: print(f"{data_label} RANSAC滤波移除了 {ransac_removed_count} 个点。保留 {len(filtered_times)} 个点。")
        else: print(f"{data_label} RANSAC滤波未移除点。保留 {len(filtered_times)} 个点。")
        if len(filtered_times) < 2: raise ValueError(f"{data_label}: RANSAC滤波后剩余数据点少于2个。无法继续。")
        return {
            'timestamps': filtered_times, 'positions': filtered_utm_positions, 
            'utm_zone': f"{utm_zone_number}{'S' if 'south' in utm_hemisphere else 'N'}", 'projector': projector
        }
    except FileNotFoundError: raise ValueError(f"{data_label} 文件未找到: {txt_path}")
    except Exception as e:
        print(f"加载/处理 {data_label} 数据 ({txt_path}) 出错:")
        traceback.print_exc() 
        raise ValueError(f"{data_label} 数据处理失败: {str(e)}")

def utm_to_wgs84(utm_points: np.ndarray, projector: Proj) -> np.ndarray:
    """将UTM坐标转换为WGS84经纬高。"""
    if utm_points.shape[1] != 3: raise ValueError("UTM点必须是Nx3数组 (X, Y, Z)")
    if not isinstance(projector, Proj): raise TypeError("projector必须是pyproj.Proj实例")
    lons, lats = projector(utm_points[:, 0], utm_points[:, 1], inverse=True)
    return np.column_stack((lons, lats, utm_points[:, 2]))

# ----------------------------
# 时间对齐和变换计算
# ----------------------------
def estimate_time_offset(slam_times: np.ndarray, gps_times: np.ndarray, max_samples: int) -> float:
    """通过互相关估计时钟偏移。"""
    if len(slam_times) < 2 or len(gps_times) < 2:
        print("警告: SLAM或GPS时间序列过短 (<2)，无法可靠估计时间偏移。返回偏移0。")
        return 0.0
    num_samples = min(max_samples, len(slam_times), len(gps_times))
    if num_samples < 2: print("警告: 用于相关的有效样本数 < 2。返回偏移0。"); return 0.0
    slam_sample_times = np.linspace(slam_times.min(), slam_times.max(), num_samples)
    gps_sample_times = np.linspace(gps_times.min(), gps_times.max(), num_samples)
    slam_norm = (slam_sample_times - np.mean(slam_sample_times))
    gps_norm = (gps_sample_times - np.mean(gps_sample_times))
    slam_std, gps_std = np.std(slam_norm), np.std(gps_norm)
    if slam_std < 1e-9 or gps_std < 1e-9: 
         print("警告: 采样后时间戳标准差过小。相关性可能不稳定。返回偏移0。")
         return 0.0
    slam_norm /= slam_std; gps_norm /= gps_std
    corr = np.correlate(slam_norm, gps_norm, mode='full')
    lag = corr.argmax() - len(slam_norm) + 1 
    dt_resampled = (slam_sample_times[-1] - slam_sample_times[0]) / (num_samples - 1) if num_samples > 1 else 0.0
    if dt_resampled == 0.0 and num_samples > 1 : print("警告: 重采样的时间分辨率为零。")
    offset = lag * dt_resampled 
    print(f"估计的初始时间偏移: {offset:.3f} 秒 (正值表示GPS时间戳晚于SLAM)")
    return offset

def dynamic_time_alignment(slam_data: Dict[str, np.ndarray],
                           gps_data_source: Dict[str, np.ndarray], 
                           time_align_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """动态地将GPS数据与SLAM时间戳对齐，处理中断。"""
    slam_times, gps_times, gps_positions = slam_data['timestamps'], gps_data_source['timestamps'], gps_data_source['positions']
    max_corr_samples, max_gps_gap_threshold = time_align_config['max_samples_for_corr'], time_align_config['max_gps_gap_threshold']
    n_slam, n_gps = len(slam_times), len(gps_times)
    aligned_gps_full, valid_mask = np.full((n_slam, 3), np.nan), np.zeros(n_slam, dtype=bool)       
    if n_slam == 0 or n_gps < 2: 
        print(f"警告: SLAM时间戳为空 ({n_slam}) 或有效GPS点不足 ({n_gps} < 2)。跳过时间对齐。")
        return aligned_gps_full, valid_mask

    offset = estimate_time_offset(slam_times, gps_times, max_corr_samples)
    adjusted_gps_times = gps_times + offset 
    try:
        sorted_indices = np.argsort(adjusted_gps_times)
        adjusted_gps_times_sorted, gps_positions_sorted = adjusted_gps_times[sorted_indices], gps_positions[sorted_indices]
        unique_times, unique_indices = np.unique(adjusted_gps_times_sorted, return_index=True)
        n_unique_gps = len(unique_times)
        if n_unique_gps < 2: 
             print("警告: 调整和排序后唯一GPS时间戳少于2个。无法进行插值。")
             return aligned_gps_full, valid_mask
        if n_unique_gps < n_gps: 
            print(f"警告: 移除了 {n_gps - n_unique_gps} 个重复的GPS时间戳。")
            adjusted_gps_times_sorted, gps_positions_sorted = unique_times, gps_positions_sorted[unique_indices]

        time_diffs = np.diff(adjusted_gps_times_sorted)
        gap_indices = np.where(time_diffs > max_gps_gap_threshold)[0]
        segment_starts = [0] + (gap_indices + 1).tolist()
        segment_ends = gap_indices.tolist() + [n_unique_gps - 1] 
        num_segments = len(segment_starts)
        print(f"检测到 {num_segments - 1} 个GPS时间中断 (间隙 > {max_gps_gap_threshold:.1f}s)。处理 {num_segments} 个段。")
        total_valid_points = 0
        for i in range(num_segments):
            start_idx, end_idx = segment_starts[i], segment_ends[i]
            segment_len = end_idx - start_idx + 1
            if segment_len < 2 : continue
            kind = 'cubic' if segment_len >= 4 else 'linear'
            segment_times, segment_positions = adjusted_gps_times_sorted[start_idx:end_idx+1], gps_positions_sorted[start_idx:end_idx+1]
            if not np.all(np.diff(segment_times) > 1e-9): 
                 print(f"警告: 段 {i+1} 时间戳在去重后非严格递增。跳过此段。")
                 continue
            try:
                interp_func_segment = interp1d(segment_times, segment_positions, axis=0, kind=kind, bounds_error=False, fill_value=np.nan)
            except ValueError as e:
                print(f"警告: 为段 {i+1} ({kind}, {segment_len} 点) 创建插值器失败: {e}。跳过。")
                continue
            epsilon = 1e-9
            slam_indices_in_segment = np.where((slam_times >= segment_times[0]-epsilon) & (slam_times <= segment_times[-1]+epsilon))[0]
            if len(slam_indices_in_segment) > 0:
                interpolated_positions = interp_func_segment(slam_times[slam_indices_in_segment])
                aligned_gps_full[slam_indices_in_segment] = interpolated_positions
                non_nan_mask_segment = ~np.isnan(interpolated_positions).any(axis=1)
                valid_indices_in_segment = slam_indices_in_segment[non_nan_mask_segment]
                valid_mask[valid_indices_in_segment] = True
                total_valid_points += len(valid_indices_in_segment)
        print(f"分段插值完成: 从 {n_slam} 个SLAM时间戳生成了 {total_valid_points} 个有效的对齐GPS位置。")
        if total_valid_points == 0: print("警告: 未生成有效的对齐GPS位置。检查时间重叠、间隙阈值或数据质量。")
        return aligned_gps_full, valid_mask
    except ValueError as e: 
        print(f"时间对齐或分段插值过程中出错: {e}。")
        traceback.print_exc()
        return np.full((n_slam, 3), np.nan), np.zeros(n_slam, dtype=bool)

def compute_sim3_transform_robust(src: np.ndarray, dst: np.ndarray,
                                  min_samples: int, residual_threshold: float,
                                  max_trials: int, min_inliers_needed: int,
                                  point_description: str = "点") -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """使用RANSAC从src到dst稳健地估计Sim3变换。"""
    n_points = src.shape[0]
    if n_points < min_samples:
        print(f"错误: Sim3 RANSAC: 输入点数不足 ({n_points} 来自 {point_description}) 进行RANSAC (至少需要 {min_samples} 个)。")
        return None, None, None
    if src.shape != dst.shape:
         print(f"错误: Sim3 RANSAC: 源点和目标点 ({point_description}) 数量或维度不匹配 ({src.shape} vs {dst.shape})。")
         return None, None, None

    best_inlier_mask, max_inliers = None, -1
    print(f"  在 {n_points} 个{point_description}上开始Sim3 RANSAC (阈值={residual_threshold}m, 尝试次数={max_trials}, 最小样本数={min_samples})...")
    for _ in range(max_trials):
        indices = np.random.choice(n_points, min_samples, replace=False)
        src_sample, dst_sample = src[indices], dst[indices]
        R_trial, t_trial, scale_trial = compute_sim3_transform(src_sample, dst_sample)
        if R_trial is None: continue 
        src_transformed_trial = scale_trial * (src @ R_trial.T) + t_trial
        residuals = np.linalg.norm(src_transformed_trial - dst, axis=1)
        inlier_mask_trial = residuals < residual_threshold
        num_inliers_trial = np.sum(inlier_mask_trial)
        if num_inliers_trial > max_inliers:
            max_inliers, best_inlier_mask = num_inliers_trial, inlier_mask_trial
    print(f"  Sim3 RANSAC 完成: 找到最大内点数: {max_inliers}/{n_points}。")
    if max_inliers < min_inliers_needed:
        print(f"错误: Sim3 RANSAC: 找到的最佳内点数不足 ({max_inliers} 来自 {point_description}) (需要 {min_inliers_needed} 个)。无法计算可靠的Sim3。")
        return None, None, None
    print(f"  使用 {max_inliers} 个内点重新计算最终Sim3变换...")
    src_inliers, dst_inliers = src[best_inlier_mask], dst[best_inlier_mask]
    final_R, final_t, final_scale = compute_sim3_transform(src_inliers, dst_inliers)
    if final_R is None: 
        print(f"错误: Sim3 RANSAC: 使用 {max_inliers} 个内点计算最终变换失败。")
        return None, None, None
    print(f"  最终Sim3参数 (来自内点): 尺度={final_scale:.4f}")
    return final_R, final_t, final_scale

def compute_sim3_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """使用Umeyama方法计算从src到dst的最优Sim3变换 (R, t, s)。"""
    n_points = src.shape[0]
    if n_points < 3: return None, None, None 
    if src.shape != dst.shape or src.shape[1] != 3:
         print(f"错误: Sim3计算: 源/目标点维度不正确 (应为Nx3)。")
         return None, None, None
    try:
        src_centroid, dst_centroid = np.mean(src, axis=0), np.mean(dst, axis=0)
        src_centered, dst_centered = src - src_centroid, dst - dst_centroid
        H = src_centered.T @ dst_centered
        U, S_svd, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T 
        if np.linalg.det(R) < 0: 
            Vt_copy = Vt.copy(); Vt_copy[-1, :] *= -1; R = Vt_copy.T @ U.T
        var_src = np.sum(np.sum(src_centered**2, axis=1)) / n_points 
        trace_S_diag_R_det = np.sum(S_svd * np.diag(np.eye(3) @ np.diag([1,1,np.linalg.det(R)]))) 
        if var_src < 1e-12: 
             print("警告: 源点集方差接近零。尺度计算可能不可靠。默认尺度为1.0。")
             scale = 1.0
        else:
            scale = trace_S_diag_R_det / (n_points * var_src) 
            if scale <= 1e-6: print(f"警告: 计算出的尺度非常小 ({scale:.2e})。重置为1.0。"); scale = 1.0
        t = dst_centroid - scale * (R @ src_centroid)
        return R, t, scale
    except np.linalg.LinAlgError as e:
        print(f"错误: Sim3计算过程中的线性代数错误 (例如SVD失败): {e}")
        return None, None, None
    except Exception as e: 
        print(f"错误: Sim3计算过程中的未知错误: {e}")
        traceback.print_exc()
        return None, None, None

def transform_trajectory(positions: np.ndarray, quaternions: np.ndarray,
                        R_mat: np.ndarray, t_vec: np.ndarray, scale_val: float) -> Tuple[np.ndarray, np.ndarray]:
    """将Sim3变换应用于整个轨迹（位置和方向）。"""
    trans_pos = scale_val * (positions @ R_mat.T) + t_vec 
    R_sim3_rot = Rotation.from_matrix(R_mat)
    trans_quat_list = [ (R_sim3_rot * Rotation.from_quat(q)).as_quat() for q in quaternions]
    return trans_pos, np.array(trans_quat_list)

# --- 可视化和评估系统 ---
def plot_results(original_pos: np.ndarray,
                 sim3_pos_ekf_input: np.ndarray, # This is the Sim3 aligned trajectory
                 corrected_pos: np.ndarray,      # This is the EKF fused trajectory
                 gps_pos: np.ndarray,
                 ground_truth_gps_raw_utm: Optional[np.ndarray],
                 aligned_ref_gps_for_error_plot: np.ndarray,
                 valid_indices_for_error_plot: np.ndarray,
                 slam_times: np.ndarray,
                 ekf_errors: Optional[np.ndarray] = None,
                 sim3_errors_for_plot: Optional[np.ndarray] = None, # ADDED: Sim3 errors
                 error_reference_name: str = "GPS Reference"
                ) -> None:
    """Enhanced visualization: trajectories, GPS, ground truth, errors, with toggleable display."""
    # Try to set a font that supports Chinese characters, fallback if not available
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'WenQuanYi Zen Hei', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False 
    except:
        print("Warning: Failed to set Chinese font. Labels might not display correctly.")

    fig = plt.figure(figsize=(18, 12))
    # Main title in English as requested for the module, but can be adapted
    fig.suptitle(f"SLAM-GPS Trajectory Alignment and Fusion Results", fontsize=16)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.2, 1, 1], height_ratios=[1,1], wspace=0.3, hspace=0.3)

    ax_check_buttons = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax3d = fig.add_subplot(gs[0, 2], projection='3d')
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_err_time = fig.add_subplot(gs[1, 2]) # This is the "Error over time" plot

    # --- 1. 2D Trajectory Comparison (XY Plane) ---
    line_orig, = ax1.plot(original_pos[:, 0], original_pos[:, 1], 'b--', alpha=0.6, linewidth=1, label='Original SLAM')
    line_sim3, = ax1.plot(sim3_pos_ekf_input[:, 0], sim3_pos_ekf_input[:, 1], 'm:', alpha=0.7, linewidth=1, label='Sim3 Aligned (EKF Input)')
    line_ekf, = ax1.plot(corrected_pos[:, 0], corrected_pos[:, 1], 'g-', linewidth=1.5, label='EKF Fused')
    scatter_gps = ax1.scatter(gps_pos[:, 0], gps_pos[:, 1], c='r', marker='.', s=30, label='Primary GPS (Filtered)')

    scatter_gt_gps_traj_2d = None
    if ground_truth_gps_raw_utm is not None and ground_truth_gps_raw_utm.shape[0] > 0:
        scatter_gt_gps_traj_2d, = ax1.plot(ground_truth_gps_raw_utm[:, 0], ground_truth_gps_raw_utm[:, 1],
                                             c='darkcyan', linewidth=1.5,label='Ground Truth GNSS Traj.', alpha=0.5, zorder=4)

    scatter_aligned_ref_gps_2d = None
    step = max(1, len(valid_indices_for_error_plot) // 100) if len(valid_indices_for_error_plot) > 0 else 1
    if len(valid_indices_for_error_plot) > 0 and aligned_ref_gps_for_error_plot.ndim == 2 and aligned_ref_gps_for_error_plot.shape[0] > 0:
        points_to_scatter = aligned_ref_gps_for_error_plot[::step]
        if points_to_scatter.shape[0] > 0:
            scatter_aligned_ref_gps_2d = ax1.scatter(points_to_scatter[:, 0], points_to_scatter[:, 1],
                                        facecolors='none', edgecolors='orange', marker='o', s=40,
                                        label=f'Aligned Pts for Err ({error_reference_name})') # Label in English
    ax1.set_title('Trajectory Comparison (X-Y Plane)') # English Title
    ax1.set_xlabel('X (meters)'); ax1.set_ylabel('Y (meters)') # English Labels
    ax1.grid(True); ax1.axis('equal'); ax1.legend(loc='best')

    # --- 2. 3D Trajectory Comparison ---
    line_orig_3d, = ax3d.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 'b--', alpha=0.6, lw=1, label='Original SLAM')
    line_sim3_3d, = ax3d.plot(sim3_pos_ekf_input[:, 0], sim3_pos_ekf_input[:, 1], sim3_pos_ekf_input[:, 2], 'm:', alpha=0.7, lw=1, label='Sim3 Aligned (EKF Input)')
    line_ekf_3d, = ax3d.plot(corrected_pos[:, 0], corrected_pos[:, 1], corrected_pos[:, 2], 'g-', lw=1.5, label='EKF Fused')
    scatter_gps_3d = ax3d.scatter(gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2], c='r', marker='x', s=30, label='Primary GPS (Filtered)')

    scatter_gt_gps_traj_3d = None
    if ground_truth_gps_raw_utm is not None and ground_truth_gps_raw_utm.shape[0] > 0:
        scatter_gt_gps_traj_3d = ax3d.scatter(ground_truth_gps_raw_utm[:, 0], ground_truth_gps_raw_utm[:, 1], ground_truth_gps_raw_utm[:, 2],
                                              c='darkcyan', marker='P', s=35, label='Ground Truth GNSS Traj.', alpha=0.7, zorder=4)
    scatter_aligned_ref_gps_3d = None
    if len(valid_indices_for_error_plot) > 0 and aligned_ref_gps_for_error_plot.ndim == 2 and aligned_ref_gps_for_error_plot.shape[0] > 0 and aligned_ref_gps_for_error_plot.shape[1] ==3:
        points_to_scatter_3d = aligned_ref_gps_for_error_plot[::step]
        if points_to_scatter_3d.shape[0] > 0:
            scatter_aligned_ref_gps_3d = ax3d.scatter(points_to_scatter_3d[:, 0], points_to_scatter_3d[:, 1], points_to_scatter_3d[:, 2],
                                            facecolors='none', edgecolors='orange', marker='o', s=40,
                                            label=f'Aligned Pts for Err ({error_reference_name})') # Label in English
    ax3d.set_title('Trajectory Comparison (3D)') # English Title
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)') # English Labels
    ax3d.legend(loc='best')
    try:
        all_traj_pts_3d = [p for p in [original_pos, sim3_pos_ekf_input, corrected_pos, gps_pos, ground_truth_gps_raw_utm] if p is not None and p.ndim==2 and p.shape[0]>0 and p.shape[1]==3]
        if all_traj_pts_3d:
            plot_center_data_3d = np.vstack(all_traj_pts_3d)
            max_range_val_3d = np.array([plot_center_data_3d[:,0].max()-plot_center_data_3d[:,0].min(),
                                     plot_center_data_3d[:,1].max()-plot_center_data_3d[:,1].min(),
                                     plot_center_data_3d[:,2].max()-plot_center_data_3d[:,2].min()]).max()/2.0 * 1.1
            if max_range_val_3d < 1.0: max_range_val_3d = 5.0
            center_priority_3d = [corrected_pos, sim3_pos_ekf_input, original_pos, gps_pos, ground_truth_gps_raw_utm]
            final_center_data_3d = next((cp for cp in center_priority_3d if cp is not None and cp.ndim==2 and cp.shape[0]>0 and cp.shape[1]==3), plot_center_data_3d)
            mid_x, mid_y, mid_z = np.median(final_center_data_3d[:,0]), np.median(final_center_data_3d[:,1]), np.median(final_center_data_3d[:,2])
            ax3d.set_xlim(mid_x - max_range_val_3d, mid_x + max_range_val_3d)
            ax3d.set_ylim(mid_y - max_range_val_3d, mid_y + max_range_val_3d)
            ax3d.set_zlim(mid_z - max_range_val_3d, mid_z + max_range_val_3d)
    except Exception as e_3d_scale: print(f"Warning (plot_results): 3D plot auto-scaling failed: {e_3d_scale}")


    # --- CheckButtons for Toggling Visibility ---
    lines_to_toggle_map = {
        'Original SLAM': (line_orig, line_orig_3d),
        'Sim3 Aligned': (line_sim3, line_sim3_3d),
        'EKF Fused': (line_ekf, line_ekf_3d),
        'Primary GPS': (scatter_gps, scatter_gps_3d),
        f'Aligned Pts (Err vs {error_reference_name})': (scatter_aligned_ref_gps_2d, scatter_aligned_ref_gps_3d),
    }
    if scatter_gt_gps_traj_2d is not None and scatter_gt_gps_traj_3d is not None:
        lines_to_toggle_map['Ground Truth GNSS Traj.'] = (scatter_gt_gps_traj_2d, scatter_gt_gps_traj_3d)

    checkbox_labels = [k for k, v_tuple in lines_to_toggle_map.items() if v_tuple[0] is not None or v_tuple[1] is not None]
    initial_visibility = [True] * len(checkbox_labels)

    check = CheckButtons(ax=ax_check_buttons, labels=checkbox_labels, actives=initial_visibility,
                         label_props={'fontsize': [9]}, frame_props={'edgecolor': ['gray']}, check_props={'linewidth': [1.5]})
    ax_check_buttons.set_title("Show/Hide Layers", fontsize=10) # English Title

    def toggle_line_visibility_combined(label):
        artist_tuple = lines_to_toggle_map.get(label)
        if not artist_tuple: return
        artist_2d, artist_3d = artist_tuple

        current_visibility = None
        if artist_2d:
            current_visibility = artist_2d.get_visible()
            artist_2d.set_visible(not current_visibility)

        if artist_3d:
            if current_visibility is None and artist_2d is None:
                 current_visibility = artist_3d.get_visible()
            artist_3d.set_visible(not current_visibility if current_visibility is not None else not artist_3d.get_visible())

        ax1.legend(loc='best'); ax3d.legend(loc='best')
        fig.canvas.draw_idle()
    check.on_clicked(toggle_line_visibility_combined)
    fig._widgets_store = [check]

    # --- 3. Final Error Distribution (Histogram) ---
    ax_hist.set_title(f"Position Error Distribution (vs {error_reference_name})") # English Title
    if ekf_errors is not None and len(ekf_errors) > 0:
        mean_err, std_err, max_err_val, median_err = np.mean(ekf_errors), np.std(ekf_errors), np.max(ekf_errors), np.median(ekf_errors)
        rmse = np.sqrt(np.mean(ekf_errors**2))
        ax_hist.hist(ekf_errors, bins=30, alpha=0.75, color='purple', label='Fused Error') # English Label
        ax_hist.axvline(mean_err, color='red', ls='--', lw=1, label=f'Mean (EKF): {mean_err:.2f}m')
        ax_hist.axvline(median_err, color='orange', ls='--', lw=1, label=f'Median (EKF): {median_err:.2f}m')
        ax_hist.axvline(rmse, color='cyan', ls=':', lw=1, label=f'RMSE (EKF): {rmse:.2f}m')
        ax_hist.set_xlabel('Absolute Position Error (meters)'); ax_hist.set_ylabel('Frequency') # English Labels
        ax_hist.legend(); ax_hist.grid(axis='y', linestyle=':')
    else:
        ax_hist.text(0.5, 0.5, "No valid EKF error data", ha='center', va='center', fontsize=12, transform=ax_hist.transAxes) # English Text
        ax_hist.set_xlabel('Error (m)'); ax_hist.set_ylabel('Frequency') # English Labels

    # --- 4. Error Over Time Plot (MODIFIED SECTION) ---
    ax_err_time.set_title(f'Error Over Time (vs {error_reference_name})') # English Title
    ax_err_time.set_xlabel('Relative Time (seconds)'); ax_err_time.set_ylabel('Error (meters)') # English Labels
    ax_err_time.grid(True)
    
    plotted_ekf_error = False
    plotted_sim3_error = False

    if len(valid_indices_for_error_plot) > 0 and len(slam_times) > 0:
        # Ensure corrected_pos and sim3_pos_ekf_input align with slam_times for indexing
        if corrected_pos.shape[0] == len(slam_times) and sim3_pos_ekf_input.shape[0] == len(slam_times):
            valid_timestamps_for_error = slam_times[valid_indices_for_error_plot]
            
            if len(valid_timestamps_for_error) > 0:
                relative_time = valid_timestamps_for_error - valid_timestamps_for_error[0]

                # Plot EKF Fused Error
                if ekf_errors is not None and len(ekf_errors) == len(relative_time):
                    ax_err_time.plot(relative_time, ekf_errors, 'g-', lw=1.5, alpha=0.9, label='Fused Error') # English Label
                    plotted_ekf_error = True
                else:
                    print("Warning (Error Over Time): EKF error data mismatch or unavailable for plotting.")

                # Plot Sim3 Aligned Error
                if sim3_errors_for_plot is not None and len(sim3_errors_for_plot) == len(relative_time):
                    ax_err_time.plot(relative_time, sim3_errors_for_plot, 'm--', lw=1, alpha=0.7, label='Slam Error') # English Label
                    plotted_sim3_error = True
                else:
                    print("Warning (Error Over Time): Sim3 error data mismatch or unavailable for plotting.")

                if plotted_ekf_error or plotted_sim3_error:
                    all_plotted_errors = []
                    if plotted_ekf_error: all_plotted_errors.extend(ekf_errors)
                    if plotted_sim3_error: all_plotted_errors.extend(sim3_errors_for_plot)
                    
                    if all_plotted_errors:
                        min_plot_err_val = np.min(all_plotted_errors)
                        max_plot_err_val = np.max(all_plotted_errors)
                        y_bottom_margin = 0.1 * (max_plot_err_val - min_plot_err_val) if max_plot_err_val > min_plot_err_val else 0.1 * max_plot_err_val
                        min_plot_y_limit = max(0, min_plot_err_val - y_bottom_margin)
                        ax_err_time.set_ylim(bottom=min_plot_y_limit)
                    ax_err_time.legend(loc='best') # Add legend
                else:
                    ax_err_time.text(0.5, 0.5, "Timestamp/error count mismatch for plot", ha='center', va='center', transform=ax_err_time.transAxes) # English Text
            else:
                 ax_err_time.text(0.5, 0.5, "No valid timestamps for error plot after filtering", ha='center', va='center', transform=ax_err_time.transAxes) # English Text
        else:
            ax_err_time.text(0.5, 0.5, "SLAM timestamp/pose count mismatch for plot", ha='center', va='center', transform=ax_err_time.transAxes) # English Text
    else:
        ax_err_time.text(0.5, 0.5, "No valid error/timestamp data for plot", ha='center', va='center', transform=ax_err_time.transAxes) # English Text

    fig.tight_layout(rect=[0.08, 0.03, 1, 0.95]); fig.subplots_adjust(top=0.92)
    plt.show()


# --- GUI 文件选择函数 ---
def select_file_dialog(title: str, filetypes: List[Tuple[str, str]]) -> str:
    root = tk.Tk(); root.withdraw(); file_path = filedialog.askopenfilename(title=title, filetypes=filetypes); root.destroy(); return file_path or ""
def select_slam_file() -> str: return select_file_dialog("选择SLAM轨迹文件 (TUM格式)", [("文本文件", "*.txt"), ("所有文件", "*.*")])
def select_gps_file(prompt_title: str = "选择主GPS数据文件 (ts lat lon alt ...)") -> str: return select_file_dialog(prompt_title, [("文本文件", "*.txt"), ("CSV文件", "*.csv"),("所有文件", "*.*")])
def select_ground_truth_gps_file() -> str: return select_gps_file(prompt_title="选择GNSS真值数据文件 (ts lat lon alt ...)")

# ----------------------------
# EKF 实现
# ----------------------------
class ExtendedKalmanFilter:
    def __init__(self, initial_pos: np.ndarray, initial_quat: np.ndarray, config_params: Dict[str, Any]): 
        if not (initial_pos.shape == (3,) and initial_quat.shape == (4,)): raise ValueError("EKF初始化: 初始位姿维度不正确。")
        ekf_cfg = config_params 
        self.state = np.concatenate([initial_pos, self.normalize_quaternion(initial_quat)]).astype(float)
        self.cov = np.diag(ekf_cfg['initial_cov_diag']).astype(float)
        self.Q_per_sec = np.diag(ekf_cfg['process_noise_diag']).astype(float) 
        self.R = np.diag(ekf_cfg['meas_noise_diag']).astype(float)           
        if self.state.shape!=(7,) or self.cov.shape!=(7,7) or self.Q_per_sec.shape!=(7,7) or self.R.shape!=(3,3):
            raise ValueError("EKF初始化: 状态/协方差/噪声矩阵维度不正确。")
        
        self.gnss_available_prev = None 
        self.gnss_update_weight = 0.0
        self.original_transition_steps = max(1, int(ekf_cfg.get('transition_steps', 10)))
        self.current_transition_steps = self.original_transition_steps # 可被外部临时修改
        self.weight_delta = 1.0 # 将在process_step中根据current_transition_steps计算
        self._last_predicted_state_for_blending = self.state.copy()

    @staticmethod
    def normalize_quaternion(q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q)
        return q/norm if norm > 1e-9 else np.array([0.0,0.0,0.0,1.0])

    def _predict(self, current_state: np.ndarray, current_cov: np.ndarray, 
                 slam_motion_update: Tuple[np.ndarray, np.ndarray], delta_time: float) -> Tuple[np.ndarray, np.ndarray]:
        prev_pos, prev_quat = current_state[:3], current_state[3:]
        prev_rot = Rotation.from_quat(prev_quat)
        delta_pos_local, delta_quat_val = slam_motion_update 
        delta_rot = Rotation.from_quat(delta_quat_val)
        predicted_pos = prev_pos + prev_rot.apply(delta_pos_local)
        predicted_rot = prev_rot * delta_rot
        predicted_quat = self.normalize_quaternion(predicted_rot.as_quat())
        predicted_state = np.concatenate([predicted_pos, predicted_quat])
        dt_adj = max(abs(delta_time), 1e-6) 
        Q_dt = self.Q_per_sec * dt_adj 
        predicted_covariance = current_cov + Q_dt 
        return predicted_state, (predicted_covariance + predicted_covariance.T) / 2.0

    def _update(self, predicted_state: np.ndarray, predicted_covariance: np.ndarray,
                gps_pos_meas: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]: 
        if gps_pos_meas.shape != (3,) or np.isnan(gps_pos_meas).any(): return None, None 
        H_jac = np.zeros((3,7)); H_jac[0,0]=1; H_jac[1,1]=1; H_jac[2,2]=1 
        try:
            pred_meas = predicted_state[:3]; innovation = gps_pos_meas - pred_meas  
            S_innov_cov = H_jac @ predicted_covariance @ H_jac.T + self.R 
            S_innov_cov = (S_innov_cov + S_innov_cov.T) / 2.0 
            try: S_inv = np.linalg.inv(S_innov_cov)
            except np.linalg.LinAlgError: S_inv = np.linalg.pinv(S_innov_cov); print(f"警告 (EKF更新): S矩阵接近奇异。使用伪逆。")
            K_gain = predicted_covariance @ H_jac.T @ S_inv
            updated_state = predicted_state + K_gain @ innovation
            updated_state[3:] = self.normalize_quaternion(updated_state[3:]) 
            I_mat = np.eye(7) 
            updated_covariance = (I_mat - K_gain @ H_jac) @ predicted_covariance @ (I_mat - K_gain @ H_jac).T + K_gain @ self.R @ K_gain.T
            return updated_state, (updated_covariance + updated_covariance.T) / 2.0
        except np.linalg.LinAlgError as e: print(f"警告 (EKF更新): 线性代数错误: {e}。跳过更新。"); return None,None
        except Exception as e: print(f"警告 (EKF更新): 未知错误: {e}。跳过更新。"); traceback.print_exc(); return None,None

    def process_step(self, slam_motion_update: Tuple[np.ndarray, np.ndarray],
                     gps_measurement: Optional[np.ndarray], gnss_is_available: bool, 
                     delta_time: float, 
                     override_transition_steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        effective_transition_steps = override_transition_steps if override_transition_steps is not None else self.current_transition_steps
        self.weight_delta = 1.0 / effective_transition_steps if effective_transition_steps > 0 else 1.0

        pred_state_before_update, pred_cov_before_update = self._predict(self.state, self.cov, slam_motion_update, delta_time)
        self._last_predicted_state_for_blending = pred_state_before_update.copy()
        updated_state_val, updated_cov_val, update_successful = None, None, False
        
        if gnss_is_available and gps_measurement is not None:
            res = self._update(pred_state_before_update, pred_cov_before_update, gps_measurement)
            if res[0] is not None: updated_state_val, updated_cov_val, update_successful = res[0], res[1], True

        just_recovered = gnss_is_available and (self.gnss_available_prev == False)
        if gnss_is_available: 
            if just_recovered or effective_transition_steps == 0: # 如果是硬更新，权重直接为1
                 self.gnss_update_weight = 1.0 if effective_transition_steps == 0 else self.weight_delta
            elif self.gnss_update_weight < 1.0: 
                self.gnss_update_weight = min(1.0, self.gnss_update_weight + self.weight_delta)
        else: self.gnss_update_weight = 0.0 

        final_fused_state, final_fused_cov = pred_state_before_update, pred_cov_before_update
        if gnss_is_available and update_successful:
            if self.gnss_update_weight < 1.0 and effective_transition_steps > 0 : #仅当需要平滑过渡时
                w = self.gnss_update_weight
                smooth_pos = (1.0-w)*self._last_predicted_state_for_blending[:3] + w*updated_state_val[:3]
                smooth_quat = quaternion_nlerp(self._last_predicted_state_for_blending[3:], updated_state_val[3:], w)
                final_fused_state = np.concatenate([smooth_pos, smooth_quat])
                final_fused_cov = updated_cov_val 
            else: final_fused_state, final_fused_cov = updated_state_val, updated_cov_val
        
        self.state, self.cov = final_fused_state.copy(), final_fused_cov.copy()
        self.gnss_available_prev = gnss_is_available 
        return self.state, self.cov, pred_state_before_update, pred_cov_before_update

# ----------------------------
# RTS 平滑器实现
# ----------------------------
def rts_smoother_segment(states_filt_segment: List[np.ndarray], covs_filt_segment: List[np.ndarray],
                         states_pred_segment: List[np.ndarray], covs_pred_segment: List[np.ndarray]
                         ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    n_segment = len(states_filt_segment)
    if n_segment == 0: return [], []
    smoothed_states, smoothed_covs = [None]*n_segment, [None]*n_segment
    smoothed_states[-1], smoothed_covs[-1] = states_filt_segment[-1].copy(), covs_filt_segment[-1].copy()

    for k in range(n_segment - 2, -1, -1):
        P_k_k_filt, P_k_plus_1_k_pred = covs_filt_segment[k], covs_pred_segment[k+1] 
        try:
            inv_P_pred = np.linalg.inv(P_k_plus_1_k_pred)
            A_k = P_k_k_filt @ inv_P_pred # F=I
        except np.linalg.LinAlgError:
            print(f"警告 (RTS平滑器): 矩阵 P_pred[{k+1}|{k}] 在求逆时奇异。使用伪逆。")
            try: inv_P_pred = np.linalg.pinv(P_k_plus_1_k_pred); A_k = P_k_k_filt @ inv_P_pred
            except np.linalg.LinAlgError:
                print(f"错误 (RTS平滑器): P_pred[{k+1}|{k}] 伪逆也失败。跳过此步平滑。")
                smoothed_states[k], smoothed_covs[k] = states_filt_segment[k].copy(), covs_filt_segment[k].copy()
                continue
        x_k_k_filt, x_k_plus_1_smooth, x_k_plus_1_k_pred = states_filt_segment[k], smoothed_states[k+1], states_pred_segment[k+1]
        smoothed_states[k] = x_k_k_filt + A_k @ (x_k_plus_1_smooth - x_k_plus_1_k_pred)
        smoothed_states[k][3:] = ExtendedKalmanFilter.normalize_quaternion(smoothed_states[k][3:])
        P_k_plus_1_smooth = smoothed_covs[k+1]
        smoothed_covs[k] = P_k_k_filt + A_k @ (P_k_plus_1_smooth - P_k_plus_1_k_pred) @ A_k.T
        smoothed_covs[k] = (smoothed_covs[k] + smoothed_covs[k].T) / 2.0 
    return smoothed_states, smoothed_covs

# ----------------------------
# 转弯检测函数
# ----------------------------
def is_sharp_turn_in_segment(slam_quaternions_segment: List[np.ndarray], 
                               slam_timestamps_segment: List[float], 
                               yaw_rate_threshold_rad_per_sec: float) -> bool:
    """判断纯SLAM段是否包含急转弯。"""
    if len(slam_quaternions_segment) < 2: return False 
    max_observed_yaw_rate = 0.0
    for i in range(1, len(slam_quaternions_segment)):
        q1, q2 = slam_quaternions_segment[i-1], slam_quaternions_segment[i]
        t1, t2 = slam_timestamps_segment[i-1], slam_timestamps_segment[i]
        if t2 <= t1: continue
        try:
            yaw1 = Rotation.from_quat(q1).as_euler('zyx', degrees=False)[0]
            yaw2 = Rotation.from_quat(q2).as_euler('zyx', degrees=False)[0]
        except ValueError: print("警告: 转弯检测段中存在无效四元数。"); return True 
        delta_yaw = np.arctan2(np.sin(yaw2 - yaw1), np.cos(yaw2 - yaw1))
        current_yaw_rate = abs(delta_yaw / (t2 - t1))
        if current_yaw_rate > max_observed_yaw_rate: max_observed_yaw_rate = current_yaw_rate
    print(f"  纯SLAM段 ({len(slam_quaternions_segment)}点) 最大偏航角速率: {np.rad2deg(max_observed_yaw_rate):.2f} 度/秒")
    return max_observed_yaw_rate > yaw_rate_threshold_rad_per_sec

# ----------------------------
# EKF轨迹校正应用
# ----------------------------
def apply_ekf_correction(slam_data_in: Dict[str, np.ndarray], gps_data_in: Dict[str, np.ndarray], 
                         sim3_pos_initial: np.ndarray, sim3_quat_initial: np.ndarray, 
                         global_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]: 
    n_points = len(slam_data_in['timestamps'])
    if n_points == 0: return np.empty((0,3)), np.empty((0,4))
    if not (sim3_pos_initial.shape[0]==n_points and sim3_quat_initial.shape[0]==n_points):
         raise ValueError(f"Sim3变换后的轨迹点数 ({sim3_pos_initial.shape[0]}) 与SLAM时间戳数 ({n_points}) 不匹配。")

    ekf_cfg = global_config['ekf'] 
    rts_decision_cfg = global_config['rts_decision']
    
    ekf_filter = ExtendedKalmanFilter(sim3_pos_initial[0], sim3_quat_initial[0], ekf_cfg)
    # 默认情况下，我们希望EKF在GNSS恢复时进行“硬”更新，为RTS做准备
    # 如果后续判断为急转弯而不使用RTS，则会使用配置的transition_steps
    ekf_filter.current_transition_steps = 0 # 默认硬更新

    aligned_gps_for_ekf, valid_mask_for_ekf = dynamic_time_alignment(slam_data_in, gps_data_in, global_config['time_alignment'])
    ekf_filter.gnss_available_prev = valid_mask_for_ekf[0] if n_points > 0 and len(valid_mask_for_ekf) > 0 else False
    print(f"EKF已初始化。初始前一时刻GNSS状态: {ekf_filter.gnss_available_prev}")
    print(f"EKF校正: {np.sum(valid_mask_for_ekf)}/{n_points} 个SLAM点在GPS对齐时将尝试GPS更新。")

    ekf_states_filt_hist, ekf_covs_filt_hist = [ekf_filter.state.copy()], [ekf_filter.cov.copy()]
    ekf_states_pred_hist, ekf_covs_pred_hist = [ekf_filter.state.copy()], [ekf_filter.cov.copy()]

    corrected_pos_list, corrected_quat_list = np.zeros_like(sim3_pos_initial), np.zeros_like(sim3_quat_initial)
    corrected_pos_list[0], corrected_quat_list[0] = ekf_filter.state[:3].copy(), ekf_filter.state[3:].copy()

    orig_slam_pos, orig_slam_quat = slam_data_in['positions'], slam_data_in['quaternions']
    slam_timestamps = slam_data_in['timestamps']
    last_time = slam_timestamps[0]
    in_gnss_outage = not ekf_filter.gnss_available_prev
    slam_outage_start_idx = 0 if in_gnss_outage else -1

    for i in range(1, n_points):
        current_time = slam_timestamps[i]; delta_t = max(1e-6, current_time - last_time)
        slam_motion = calculate_relative_pose(orig_slam_pos[i-1], orig_slam_quat[i-1], orig_slam_pos[i], orig_slam_quat[i])
        gnss_avail = valid_mask_for_ekf[i] if i < len(valid_mask_for_ekf) else False
        gps_meas = aligned_gps_for_ekf[i] if gnss_avail and i < len(aligned_gps_for_ekf) and not np.isnan(aligned_gps_for_ekf[i]).any() else None
        if gps_meas is None: gnss_avail = False
        
        # --- 动态平滑策略决策 ---
        perform_rts = True # 默认执行RTS
        current_iter_transition_steps = 0 # 默认硬更新 (用于RTS)

        if not gnss_avail and not in_gnss_outage: # GNSS刚丢失
            in_gnss_outage = True; slam_outage_start_idx = i
            print(f"GNSS中断开始于索引 {i} (时间 {current_time:.2f}s)。")
        
        elif gnss_avail and in_gnss_outage: # GNSS刚恢复
            print(f"GNSS恢复于索引 {i} (时间 {current_time:.2f}s)。分析中断段 [{slam_outage_start_idx}-{i-1}] 的转弯情况。")
            outage_indices = range(slam_outage_start_idx, i) # 中断段的原始SLAM数据索引
            if len(outage_indices) >= 2:
                seg_quats = [orig_slam_quat[k] for k in outage_indices]
                seg_times = [slam_timestamps[k] for k in outage_indices]
                yaw_thresh_deg = rts_decision_cfg['sharp_turn_yaw_rate_threshold_deg_per_sec']
                if is_sharp_turn_in_segment(seg_quats, seg_times, np.deg2rad(yaw_thresh_deg)):
                    print(f"  检测到急转弯。此恢复点将使用EKF过渡步数，不执行RTS。")
                    perform_rts = False
                    current_iter_transition_steps = rts_decision_cfg['default_ekf_transition_steps_on_sharp_turn']
                else:
                    print(f"  路径为直线/缓弯。此恢复点将硬更新，之后执行RTS。")
                    # perform_rts 保持 True, current_iter_transition_steps 保持 0
            else:
                print(f"  中断段过短，无法分析转弯。默认执行RTS（硬更新）。")
        
        # 为当前恢复点（索引i）执行EKF步骤，使用决定的过渡步数
        fused_state, fused_cov, pred_state, pred_cov = ekf_filter.process_step(
            slam_motion, gps_meas, gnss_avail, delta_t, 
            override_transition_steps=current_iter_transition_steps if (gnss_avail and in_gnss_outage) else ekf_filter.current_transition_steps
            # 只有在GNSS恢复的那个瞬间，我们才可能覆盖transition_steps
        )
        ekf_states_filt_hist.append(fused_state.copy()); ekf_covs_filt_hist.append(fused_cov.copy())
        ekf_states_pred_hist.append(pred_state.copy()); ekf_covs_pred_hist.append(pred_cov.copy())
        corrected_pos_list[i], corrected_quat_list[i] = fused_state[:3], fused_state[3:]
        
        if gnss_avail and in_gnss_outage: # GNSS刚恢复，现在根据perform_rts决定是否执行平滑
            if perform_rts:
                # RTS段包括恢复点i本身，所以是 slam_outage_start_idx 到 i
                rts_seg_len = i - slam_outage_start_idx + 1
                if rts_seg_len > 1:
                    print(f"  对索引 [{slam_outage_start_idx}-{i}] 的 {rts_seg_len} 个状态应用RTS平滑器。")
                    seg_filt_s = ekf_states_filt_hist[slam_outage_start_idx : i + 1]
                    seg_filt_c = ekf_covs_filt_hist[slam_outage_start_idx : i + 1]
                    seg_pred_s = ekf_states_pred_hist[slam_outage_start_idx : i + 1]
                    seg_pred_c = ekf_covs_pred_hist[slam_outage_start_idx : i + 1]
                    
                    smoothed_states_seg, _ = rts_smoother_segment(seg_filt_s, seg_filt_c, seg_pred_s, seg_pred_c)
                    for k_seg in range(len(smoothed_states_seg)):
                        original_idx = slam_outage_start_idx + k_seg
                        corrected_pos_list[original_idx] = smoothed_states_seg[k_seg][:3]
                        corrected_quat_list[original_idx] = smoothed_states_seg[k_seg][3:]
                        ekf_states_filt_hist[original_idx] = smoothed_states_seg[k_seg].copy() # 更新历史记录
                    print(f"  RTS平滑已应用于索引到 {i} 的段。")
                else: print(f"  RTS段过短 (索引 {slam_outage_start_idx} 到 {i})。跳过RTS。")
            # 重置中断状态
            in_gnss_outage = False 
            slam_outage_start_idx = -1
            ekf_filter.current_transition_steps = 0 # 恢复默认硬更新，除非下次恢复时再次判断为转弯

        last_time = current_time

    if in_gnss_outage and slam_outage_start_idx != -1:
        print(f"轨迹在GNSS中断期间结束 (开始于索引 {slam_outage_start_idx})。最后一段未进行RTS平滑，因为无恢复点。")
    print("EKF校正和动态RTS平滑过程完成。")
    return corrected_pos_list, corrected_quat_list

# ----------------------------
# 主处理流程控制
# ----------------------------
def main_process_gui():
    slam_path, gps_path, ground_truth_gps_path = "", "", ""
    try:
        slam_path = select_slam_file()
        if not slam_path: print("SLAM文件选择已取消。"); return
        gps_path = select_gps_file()
        if not gps_path: print("主GPS文件选择已取消。"); return

        ground_truth_gps_data = None
        ask_for_gt = messagebox.askyesno("GNSS真值", "是否加载额外的GNSS文件作为地面真值用于误差计算？")
        if ask_for_gt:
            ground_truth_gps_path = select_ground_truth_gps_file()
            if not ground_truth_gps_path: print("GNSS真值文件选择已取消或跳过。")
            else: print(f"已选择GNSS真值文件: {ground_truth_gps_path}")
        
        print("-" * 30 + "\n文件概要:\n" + f"  SLAM: {slam_path}\n  主GPS: {gps_path}" + \
              (f"\n  GNSS真值: {ground_truth_gps_path}" if ground_truth_gps_path else "\n  GNSS真值: 未选择") + "\n" + "-"*30)

        print("步骤 1/7: 加载和预处理数据...")
        slam_data = load_slam_trajectory(slam_path)
        gps_data = load_gps_data(gps_path, data_label="主GPS", filter_config_override=CONFIG['gps_filtering_ransac'])
        print(f"  SLAM 点数: {len(slam_data['positions'])}")
        print(f"  主GPS点数 (已滤波): {len(gps_data['positions'])}")
        if ground_truth_gps_path:
            ground_truth_gps_data = load_gps_data(ground_truth_gps_path, data_label="GNSS真值", filter_config_override=CONFIG['ground_truth_gps_filtering'])
            print(f"  GNSS真值点数 (已滤波): {len(ground_truth_gps_data['positions'])}")
            if len(ground_truth_gps_data['positions']) < 2: print("警告: GNSS真值处理后点数 < 2，将不被使用。"); ground_truth_gps_data = None 
        if len(slam_data['positions']) == 0 or len(gps_data['positions']) < 2: raise ValueError("SLAM数据为空或主GPS点数不足 (<2)。无法继续。")
        print("数据加载和预处理完成。\n" + "-"*30)

        print("步骤 2/7: 为Sim3变换计算时间对齐主GPS...")
        aligned_gps_for_sim3, valid_mask_sim3 = dynamic_time_alignment(slam_data, gps_data, CONFIG['time_alignment'])
        valid_indices_all_sim3 = np.where(valid_mask_sim3)[0] 
        min_pts_sim3_ransac_sample = CONFIG['sim3_ransac']['min_samples']
        if len(valid_indices_all_sim3) < min_pts_sim3_ransac_sample:
            raise ValueError(f"用于Sim3的总有效时间同步点数 ({len(valid_indices_all_sim3)}) < RANSAC最小样本数 ({min_pts_sim3_ransac_sample}).")
        print(f"找到 {len(valid_indices_all_sim3)} 个时间同步点用于Sim3计算。")
        sim3_calc_indices = np.array([], dtype=int); point_desc_ransac = "所有有效点" 
        if len(valid_indices_all_sim3) > 0:
            valid_slam_times_sim3 = slam_data['timestamps'][valid_indices_all_sim3]
            time_diffs_sim3 = np.diff(valid_slam_times_sim3) 
            first_gap_idx_sim3 = np.where(time_diffs_sim3 > CONFIG['time_alignment']['max_gps_gap_threshold'])[0]
            end_first_segment_idx_in_valid_indices = first_gap_idx_sim3[0] if len(first_gap_idx_sim3) > 0 else len(valid_indices_all_sim3)
            first_segment_slam_indices = valid_indices_all_sim3[:end_first_segment_idx_in_valid_indices]
            if len(first_segment_slam_indices) < min_pts_sim3_ransac_sample: 
                sim3_calc_indices = valid_indices_all_sim3
                point_desc_ransac = f"所有有效点 ({len(valid_indices_all_sim3)}, 第一段太短)"
            else:
                max_dur = CONFIG['sim3_ransac']['max_initial_duration']
                segment_start_time = slam_data['timestamps'][first_segment_slam_indices[0]]
                time_lim_mask_on_segment = (slam_data['timestamps'][first_segment_slam_indices] <= segment_start_time + max_dur)
                sim3_calc_indices_timed = first_segment_slam_indices[time_lim_mask_on_segment]
                num_timed = len(sim3_calc_indices_timed)
                if num_timed < min_pts_sim3_ransac_sample: 
                    sim3_calc_indices = first_segment_slam_indices
                    point_desc_ransac = f"第一段 ({len(first_segment_slam_indices)} 点, 因点数过少移除了时间阈值)"
                else: sim3_calc_indices = sim3_calc_indices_timed; point_desc_ransac = f"初始段 (最多 {max_dur:.1f}秒, {num_timed} 点)"
        if len(sim3_calc_indices) < min_pts_sim3_ransac_sample: raise ValueError(f"用于Sim3的最终点数 ({len(sim3_calc_indices)}) < RANSAC最小样本数 ({min_pts_sim3_ransac_sample}). 无法计算Sim3。")
        print(f"使用 {len(sim3_calc_indices)} 点 ({point_desc_ransac}) 进行Sim3计算。\n" + "-"*30)

        print("步骤 3/7: 计算稳健的Sim3全局变换...")
        src_pts_sim3, dst_pts_sim3 = slam_data['positions'][sim3_calc_indices], aligned_gps_for_sim3[sim3_calc_indices] 
        R_sim3, t_sim3, scale_sim3 = compute_sim3_transform_robust(src_pts_sim3, dst_pts_sim3, CONFIG['sim3_ransac']['min_samples'], CONFIG['sim3_ransac']['residual_threshold'], CONFIG['sim3_ransac']['max_trials'], CONFIG['sim3_ransac']['min_inliers_needed'], point_desc_ransac)
        if R_sim3 is None: raise RuntimeError(f"Sim3全局变换失败 (基于 {point_desc_ransac}).")
        print("Sim3变换计算成功。")
        print("步骤 4/7: 将Sim3变换应用于完整SLAM轨迹...")
        sim3_pos, sim3_quat = transform_trajectory(slam_data['positions'], slam_data['quaternions'], R_sim3, t_sim3, scale_sim3)
        print("Sim3变换已应用。\n" + "-"*30)

        print("步骤 5/7: 应用EKF (和动态RTS平滑器) 进行轨迹融合和校正...")
        corrected_pos, corrected_quat = apply_ekf_correction(slam_data, gps_data, sim3_pos, sim3_quat, CONFIG)
        print("EKF和动态RTS平滑轨迹融合完成。\n" + "-"*30)

        print("步骤 6/7: 评估轨迹误差 (忽略前5秒, 比较物理最近邻GNSS插值点)...")
        aligned_primary_gps_for_eval_full_len, valid_mask_primary_eval = dynamic_time_alignment(slam_data, gps_data, CONFIG['time_alignment'])
        valid_slam_indices_with_primary_gps = np.where(valid_mask_primary_eval)[0]
        ekf_errors_vs_primary_filtered, sim3_errors_vs_primary_filtered = None, None
        ekf_closest_primary_pts_for_plot = np.empty((0,3))
        primary_slam_indices_post_5s = np.array([], dtype=int); primary_interpolated_candidates_post_5s = np.empty((0,3))
        if len(valid_slam_indices_with_primary_gps) > 0:
            slam_timestamps_at_valid_primary_gps = slam_data['timestamps'][valid_slam_indices_with_primary_gps]
            time_threshold = slam_data['timestamps'][0] + 5.0 
            time_filter_mask = slam_timestamps_at_valid_primary_gps > time_threshold
            primary_slam_indices_post_5s = valid_slam_indices_with_primary_gps[time_filter_mask]
            if len(primary_slam_indices_post_5s) > 0: primary_interpolated_candidates_post_5s = aligned_primary_gps_for_eval_full_len[primary_slam_indices_post_5s]
            if len(primary_slam_indices_post_5s) > 0 and len(primary_interpolated_candidates_post_5s) > 0 :
                print(f"  评估与主GPS的误差 (前5秒后, {len(primary_slam_indices_post_5s)} 点):")
                for label, traj_data_full in [("原始SLAM轨迹", slam_data['positions']), ("Sim3对齐轨迹", sim3_pos), ("EKF融合/平滑轨迹", corrected_pos)]:
                    slam_points_to_eval = traj_data_full[primary_slam_indices_post_5s]
                    if slam_points_to_eval.shape[0] > 0 and primary_interpolated_candidates_post_5s.shape[0] > 0:
                        dist_matrix = distance.cdist(slam_points_to_eval, primary_interpolated_candidates_post_5s, 'euclidean')
                        current_errors = np.min(dist_matrix, axis=1)
                        if len(current_errors) > 0: 
                            print(f"    {label:<20} -> 平均值: {np.mean(current_errors):.3f}m, 中位数: {np.median(current_errors):.3f}m, RMSE: {np.sqrt(np.mean(current_errors**2)):.3f}m")
                            if label == "EKF融合/平滑轨迹": ekf_errors_vs_primary_filtered, ekf_closest_primary_pts_for_plot = current_errors, primary_interpolated_candidates_post_5s[np.argmin(dist_matrix, axis=1)]
                            elif label == "Sim3对齐轨迹": sim3_errors_vs_primary_filtered = current_errors
                        else: print(f"    {label:<20} -> 5s截断后无误差数据。")
                    else: print(f"    {label:<20} -> 5s截断后轨迹段或候选GPS点为空。")
            else: print("警告: 5s截断后无有效的主GPS对齐点用于误差评估。")
        else: print("警告: (5s截断前)无有效的主GPS对齐点用于误差评估。")

        ekf_errors_vs_gt_filtered, sim3_errors_vs_gt_filtered = None, None
        ekf_closest_gt_pts_for_plot = np.empty((0,3))
        gt_slam_indices_post_5s = np.array([], dtype=int); gt_interpolated_candidates_post_5s = np.empty((0,3))
        if ground_truth_gps_data: 
            aligned_gt_gps_temp_full_slam_len, valid_mask_gt_temp = dynamic_time_alignment(slam_data, ground_truth_gps_data, CONFIG['time_alignment'])
            valid_slam_indices_with_gt_gps = np.where(valid_mask_gt_temp)[0] 
            if len(valid_slam_indices_with_gt_gps) > 0:
                slam_timestamps_at_valid_gt_gps = slam_data['timestamps'][valid_slam_indices_with_gt_gps]
                time_threshold_gt = slam_data['timestamps'][0] + 5.0
                time_filter_mask_gt = slam_timestamps_at_valid_gt_gps > time_threshold_gt
                gt_slam_indices_post_5s = valid_slam_indices_with_gt_gps[time_filter_mask_gt]
                if len(gt_slam_indices_post_5s) > 0: gt_interpolated_candidates_post_5s = aligned_gt_gps_temp_full_slam_len[gt_slam_indices_post_5s]
                if len(gt_slam_indices_post_5s) > 0 and len(gt_interpolated_candidates_post_5s) > 0:
                    print(f"  评估与GNSS真值的误差 (前5秒后, {len(gt_slam_indices_post_5s)} 点, 使用插值真值):")
                    for label, traj_data_full in [("原始SLAM轨迹", slam_data['positions']), ("Sim3对齐轨迹", sim3_pos), ("EKF融合/平滑轨迹", corrected_pos)]:
                        slam_points_to_eval_gt = traj_data_full[gt_slam_indices_post_5s]
                        if slam_points_to_eval_gt.shape[0] > 0 and gt_interpolated_candidates_post_5s.shape[0] > 0:
                            dist_matrix_gt = distance.cdist(slam_points_to_eval_gt, gt_interpolated_candidates_post_5s, 'euclidean')
                            current_errors_gt = np.min(dist_matrix_gt, axis=1)
                            if len(current_errors_gt) > 0:
                                print(f"    {label:<20} -> 平均值: {np.mean(current_errors_gt):.3f}m, 中位数: {np.median(current_errors_gt):.3f}m, RMSE: {np.sqrt(np.mean(current_errors_gt**2)):.3f}m")
                                if label == "EKF融合/平滑轨迹": ekf_errors_vs_gt_filtered, ekf_closest_gt_pts_for_plot = current_errors_gt, gt_interpolated_candidates_post_5s[np.argmin(dist_matrix_gt, axis=1)]
                                elif label == "Sim3对齐轨迹": sim3_errors_vs_gt_filtered = current_errors_gt
                            else: print(f"    {label:<20} -> 5s截断后无误差数据(GNSS真值)。")
                        else: print(f"    {label:<20} -> 5s截断后轨迹段或候选GNSS真值点为空。")
                else: print("警告: 5s截断后无有效的GNSS真值对齐点用于误差评估。")
            else: print("警告: (5s截断前)无有效的GNSS真值对齐点用于误差评估。")
        
        final_ekf_errors_for_plot, final_sim3_errors_for_plot = None, None
        final_aligned_ref_gps_for_plot = np.empty((0,3)); final_valid_indices_for_plot = np.array([], dtype=int)
        plot_error_ref_name = "无 (5s截断后无有效数据)"
        if ground_truth_gps_data and ekf_errors_vs_gt_filtered is not None and len(ekf_errors_vs_gt_filtered) > 0:
            final_ekf_errors_for_plot, final_aligned_ref_gps_for_plot, final_valid_indices_for_plot = ekf_errors_vs_gt_filtered, ekf_closest_gt_pts_for_plot, gt_slam_indices_post_5s
            if sim3_errors_vs_gt_filtered is not None: final_sim3_errors_for_plot = sim3_errors_vs_gt_filtered
            plot_error_ref_name = "GNSS真值 (5s后, 最近邻插值点)"
            print("绘图将主要显示与GNSS真值的误差 (5s后, 最近邻插值点)。")
        elif ekf_errors_vs_primary_filtered is not None and len(ekf_errors_vs_primary_filtered) > 0:
            final_ekf_errors_for_plot, final_aligned_ref_gps_for_plot, final_valid_indices_for_plot = ekf_errors_vs_primary_filtered, ekf_closest_primary_pts_for_plot, primary_slam_indices_post_5s
            if sim3_errors_vs_primary_filtered is not None: final_sim3_errors_for_plot = sim3_errors_vs_primary_filtered
            plot_error_ref_name = "主GPS (5s后, 最近邻插值点)"
            print("绘图将主要显示与主GPS的误差 (5s后, 最近邻插值点)。")
        else: print("5s截断后无EKF误差数据可用于主图显示。")
        print("误差评估完成。\n" + "-"*30)

        print("步骤 7/7: 保存结果和可视化...")
        if messagebox.askyesno("保存结果", "处理完成。是否保存校正后的轨迹？"):
            base_fn = slam_path.split('/')[-1].split('\\')[-1] 
            def_save_name = base_fn.replace('.txt', '_corrected_utm.txt') if '.txt' in base_fn else base_fn + '_corrected_utm.txt'
            out_path_utm = filedialog.asksaveasfilename(title="保存校正后的轨迹 (UTM, TUM格式)", defaultextension=".txt", filetypes=[("文本文件", "*.txt")], initialfile=def_save_name)
            if out_path_utm:
                out_data_utm = np.column_stack((slam_data['timestamps'], corrected_pos, corrected_quat))
                np.savetxt(out_path_utm, out_data_utm, fmt=['%.6f'] + ['%.6f']*3 + ['%.8f']*4, header="timestamp x y z qx qy qz qw (UTM)", comments='')
                print(f"  UTM轨迹已保存至: {out_path_utm}")
                try: 
                    projector = gps_data.get('projector')
                    if projector:
                        wgs84_coords = utm_to_wgs84(corrected_pos, projector) 
                        out_data_wgs84 = np.column_stack((slam_data['timestamps'], wgs84_coords, corrected_quat))
                        out_path_wgs84 = out_path_utm.replace('_utm.txt', '_wgs84.txt')
                        if out_path_wgs84 == out_path_utm: out_path_wgs84 = out_path_utm.replace('.txt', '_wgs84.txt') if '.txt' in out_path_utm else out_path_utm + '_wgs84.txt'
                        np.savetxt(out_path_wgs84, out_data_wgs84, fmt=['%.6f'] + ['%.8f','%.8f','%.3f'] + ['%.8f']*4, header="timestamp lon lat alt qx qy qz qw (WGS84)", comments='')
                        print(f"  WGS84轨迹已保存至: {out_path_wgs84}")
                    else: print("警告: 主GPS数据中未找到投影器。无法保存WGS84轨迹。")
                except Exception as e_wgs_save: print(f"保存WGS84轨迹时出错: {e_wgs_save}")
            else: print("  用户取消保存。")
        print("  生成可视化图像...")
        plot_results(slam_data['positions'], sim3_pos, corrected_pos, gps_data['positions'], 
                     ground_truth_gps_data['positions'] if ground_truth_gps_data else None, 
                     final_aligned_ref_gps_for_plot, final_valid_indices_for_plot, slam_data['timestamps'],
                     final_ekf_errors_for_plot, final_sim3_errors_for_plot, plot_error_ref_name)

    except (ValueError, FileNotFoundError, AssertionError, CRSError, np.linalg.LinAlgError, RuntimeError) as e_proc:
        error_msg = f"处理失败 ({type(e_proc).__name__}): {str(e_proc)}"
        print(f"\n错误!\n{error_msg}")
        full_error_details = f"{error_msg}\n\nSLAM文件: {slam_path or '未选择'}\n主GPS: {gps_path or '未选择'}"
        if ground_truth_gps_path: full_error_details += f"\nGNSS真值: {ground_truth_gps_path}"
        traceback.print_exc(); messagebox.showerror("处理失败", full_error_details)
    except Exception as e_unexpected:
        error_msg = f"发生意外错误: {type(e_unexpected).__name__}: {str(e_unexpected)}"
        print(f"\n严重错误!\n{error_msg}")
        full_error_details = f"{error_msg}\n\nSLAM文件: {slam_path or '未选择'}\n主GPS: {gps_path or '未选择'}"
        if ground_truth_gps_path: full_error_details += f"\nGNSS真值: {ground_truth_gps_path}"
        traceback.print_exc(); messagebox.showerror("意外错误", f"{full_error_details}\n\n详情请见控制台输出。")

if __name__ == "__main__":
    print("启动 SLAM-GPS 轨迹对齐与融合工具 (EKF + 动态RTS平滑)...")
    print("="*70 + "\n配置概览:")
    for key, cfg_section in [("主GPS RANSAC", CONFIG['gps_filtering_ransac']), ("GNSS真值 RANSAC", CONFIG['ground_truth_gps_filtering'])]:
        print(f"  {key} 滤波已启用: {cfg_section['enabled']}")
        if cfg_section['enabled']:
            mode = '滑动窗口' if cfg_section['use_sliding_window'] else '全局'
            print(f"    模式: {mode}, 阶数: {cfg_section['polynomial_degree']}, 最小样本数: {cfg_section['min_samples']}, 阈值: {cfg_section['residual_threshold_meters']}m")
            if cfg_section['use_sliding_window']: print(f"    窗口: {cfg_section['window_duration_seconds']}s, 步长因子: {cfg_section['window_step_factor']}")
    print(f"  GPS中断阈值: {CONFIG['time_alignment']['max_gps_gap_threshold']}s")
    print(f"  Sim3 RANSAC最小内点数: {CONFIG['sim3_ransac']['min_inliers_needed']}, Sim3最大初始时长: {CONFIG['sim3_ransac']['max_initial_duration']}s")
    print(f"  EKF GNSS恢复平滑步数 (急转弯时): {CONFIG['rts_decision']['default_ekf_transition_steps_on_sharp_turn']}")
    print(f"  RTS决策 - 急转弯偏航角速率阈值: {CONFIG['rts_decision']['sharp_turn_yaw_rate_threshold_deg_per_sec']} 度/秒")
    print("="*70)
    main_process_gui()
    print("\n程序运行结束。")
