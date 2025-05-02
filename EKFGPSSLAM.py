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
# 配置参数
# ----------------------------
CONFIG = {
    # EKF 参数
    "ekf": {
        # 初始状态协方差矩阵的对角线元素 (位置x,y,z, 四元数x,y,z,w) - 代表初始状态的不确定性
        "initial_cov_diag": [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01],
        # 过程噪声协方差矩阵的对角线元素 - 代表模型预测的不确定性 (SLAM漂移等)
        "process_noise_diag": [0.5, 0.5, 0.5, 0.05, 0.05, 0.05, 0.05],
        # 测量噪声协方差矩阵的对角线元素 (GPS 位置x,y,z) - 代表GPS测量的不确定性
        "meas_noise_diag": [0.3, 0.3, 0.3],
    },
    # Sim(3) 全局变换 RANSAC 参数 (之前叫 'ransac')
    "sim3_ransac": {
        # Sim(3) 估计所需的最小样本数
        "min_samples": 4,
        # RANSAC 内点判断的残差阈值 (单位：归一化空间) - 需要根据数据调整
        "residual_threshold": 0.1,
        # RANSAC 最大迭代次数
        "max_trials": 1000,
        # 接受 Sim(3) 变换所需的最小内点数
        "min_inliers_for_transform": 4, # 从 3 改为 4
    },
    # === 新增: GPS 轨迹 RANSAC 滤波参数 ===
    "gps_filtering_ransac": {
        "enabled": True, # 设置为 False 可以禁用此步骤
        # 将X,Y,Z分别拟合为时间的 N 次多项式
        "polynomial_degree": 2, # 尝试 2 或 3
         # RANSAC 拟合多项式模型所需的最小样本数 (应 >= degree + 1)
        "min_samples": 5,
        # RANSAC 内点判断的残差阈值 (单位：米)
        # *** 这个值非常关键 ***
        # 它定义了一个GPS点偏离局部拟合轨迹多远才被视为离群点。
        # 对于标准GPS，可以从 5.0 到 15.0 开始尝试。
        # 对于RTK或高精度GPS，可以设得更小 (例如 0.5 到 2.0)。
        "residual_threshold_meters": 10.0, # *** 需要根据你的GPS精度调整 ***
        # RANSAC 最大迭代次数
        "max_trials": 100,
    },
    # 时间对齐参数
    "time_alignment": {
         # 用于互相关估计时间偏移的采样点数
        "max_samples_for_corr": 500,
        # === 新增: GPS 信号中断阈值 (秒) ===
        # 如果连续两个有效GPS点的时间戳差大于此值，则认为发生中断，不进行插值
        "max_gps_gap_threshold": 5.0, # 例如，设置为 5 秒
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
            # TUM格式四元数是 qx qy qz qw, scipy 需要 x y z w, 顺序一致
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

# ----------------------------
# GPS 离群点过滤模块 (新增)
# ----------------------------
def filter_gps_outliers_ransac(times: np.ndarray, positions: np.ndarray,
                               config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 RANSAC 和多项式模型过滤 GPS 轨迹中的离群点。

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

    if len(times) < config['min_samples']:
        print(f"警告: GPS 点数 ({len(times)}) 少于 RANSAC 最小样本数 ({config['min_samples']})，跳过 GPS 离群点过滤。")
        return times, positions

    try:
        # 准备数据：时间作为特征 X (需要是 2D)，位置坐标 (x, y, z) 作为目标 y
        t_feature = times.reshape(-1, 1)

        inlier_masks = [] # 存储每个维度 (X, Y, Z) 的内点掩码
        target_names = ['X', 'Y', 'Z']

        for i in range(positions.shape[1]): # 对 X, Y, Z 维度分别拟合
            target = positions[:, i]

            # 创建多项式回归的 RANSAC 模型管道
            model = make_pipeline(
                PolynomialFeatures(degree=config['polynomial_degree']),
                RANSACRegressor(
                    min_samples=config['min_samples'],
                    residual_threshold=config['residual_threshold_meters'], # 使用米作为单位的阈值
                    max_trials=config['max_trials'],
                )
            )

            model.fit(t_feature, target)
            inlier_mask_dim = model[-1].inlier_mask_
            inlier_masks.append(inlier_mask_dim)

        # 合并掩码：要求一个点在所有维度 (X, Y, Z) 上都是内点才保留
        final_inlier_mask = np.logical_and.reduce(inlier_masks) # 使用逻辑与，要求所有维度都是内点
        num_inliers = np.sum(final_inlier_mask)
        num_outliers = len(times) - num_inliers

        if num_outliers > 0:
            print(f"GPS RANSAC 过滤: 识别并移除了 {num_outliers} 个离群点 (保留 {num_inliers} / {len(times)} 个点).")
        else:
            print("GPS RANSAC 过滤: 未发现离群点。")

        # 检查过滤后剩余的点数是否足够
        if num_inliers < config['min_samples']:
             print(f"警告: RANSAC 过滤后剩余的 GPS 点数 ({num_inliers}) 过少，可能导致后续处理失败。")

        # 返回被识别为内点的时间戳和位置
        return times[final_inlier_mask], positions[final_inlier_mask]

    except Exception as e:
        print(f"GPS RANSAC 过滤过程中发生错误: {e}. 跳过过滤步骤。")
        traceback.print_exc() # 打印详细错误信息
        return times, positions # 出错时返回原始数据

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

        # 过滤无效的经纬度值 (例如 0,0 或超出范围的值)
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
            timestamps, utm_positions, CONFIG['gps_filtering_ransac']
        )

        # 如果过滤后数据点过少，无法进行后续处理 (例如插值至少需要2个点)
        if len(filtered_times) < 2:
             raise ValueError("GPS RANSAC 过滤后剩余数据点不足 (< 2)，无法继续处理。")
        if len(filtered_times) < initial_count and np.sum(valid_gps_mask) == initial_count : # 只有RANSAC过滤了点才打印
             print(f"GPS RANSAC 过滤完成，剩余 {len(filtered_times)} 点。")
        elif len(filtered_times) < initial_count: # 如果是因为无效点被过滤
             print(f"GPS 数据过滤完成 (移除了无效点和RANSAC离群点)，剩余 {len(filtered_times)} 点。")

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

    dt_resampled = (slam_sample_times[-1] - slam_sample_times[0]) / (num_samples - 1) if num_samples > 1 else 0
    offset = lag * dt_resampled

    print(f"估计的初始时间偏移: {offset:.3f} 秒")
    return offset

# === 修改后的函数 ===
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

    if n_slam == 0 or n_gps == 0:
        print("警告：SLAM 或 GPS 时间戳为空，无法进行时间对齐。")
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
        gap_indices = np.where(time_diffs > max_gps_gap_threshold)[0] # 找到大于阈值的间隙的结束索引

        segment_starts = [0] + (gap_indices + 1).tolist() # 每个分段的起始索引 (包括第一个点0)
        segment_ends = gap_indices.tolist() + [n_unique_gps - 1] # 每个分段的结束索引 (包括最后一个点)

        print(f"检测到 {len(gap_indices)} 个 GPS 时间中断 (阈值 > {max_gps_gap_threshold:.1f}s)，将数据分为 {len(segment_starts)} 段进行插值。")

        # 4. 对每个数据段进行插值
        total_valid_points = 0
        for i in range(len(segment_starts)):
            start_idx = segment_starts[i]
            end_idx = segment_ends[i]
            segment_len = end_idx - start_idx + 1

            # 检查段内点数是否足够进行三次样条插值 (至少需要2个点)
            if segment_len < 2:
                # print(f"  跳过段 {i+1}: 点数不足 ({segment_len} < 2)")
                continue # 跳过这个太短的段

            # 提取当前段的数据
            segment_times = adjusted_gps_times_sorted[start_idx : end_idx + 1]
            segment_positions = gps_positions_sorted[start_idx : end_idx + 1]
            segment_min_time = segment_times[0]
            segment_max_time = segment_times[-1]

            # 为当前段创建插值函数
            try:
                interp_func_segment = interp1d(
                    segment_times,             # x轴: 当前段的时间
                    segment_positions,         # y轴: 当前段的位置
                    axis=0,                    # 对每一列(x,y,z)独立插值
                    kind='cubic',              # 插值方法: 三次样条
                    bounds_error=False,        # 对于超出 *当前段* 时间范围的点不报错
                    fill_value=np.nan          # 对于超出 *当前段* 范围的点填充 NaN (理论上不应发生在此处)
                )
            except ValueError as e:
                print(f"警告：为段 {i+1} 创建插值函数失败: {e}。跳过此段。")
                continue

            # 找到落入当前段时间范围内的 SLAM 时间戳索引
            slam_indices_in_segment = np.where(
                (slam_times >= segment_min_time) & (slam_times <= segment_max_time)
            )[0]

            if len(slam_indices_in_segment) > 0:
                # 对这些 SLAM 时间戳进行插值
                interpolated_positions = interp_func_segment(slam_times[slam_indices_in_segment])

                # 将插值结果填充到主数组中
                aligned_gps_full[slam_indices_in_segment] = interpolated_positions
                # 标记这些点为有效 (非NaN)
                # 需要额外检查插值结果是否为NaN (虽然理论上不应发生，除非SLAM时间正好等于段边界且插值出问题)
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
                                 max_trials: int, min_inliers_needed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    使用 RANSAC 稳健地估计源点(src)到目标点(dst)的 Sim3 变换。
    注意：此 RANSAC 在归一化坐标上运行，阈值是相对值。
    """
    if src.shape[0] < min_samples or dst.shape[0] < min_samples:
        raise ValueError(f"点数不足 ({src.shape[0]}) ，无法进行 Sim3 RANSAC (需要至少 {min_samples} 个)")
    if src.shape != dst.shape:
         raise ValueError("Sim3 RANSAC：源点和目标点数量必须一致")

    # 数据归一化：移至质心，并按平均距离缩放（有助于设定相对阈值）
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    src_norm_factor = np.mean(np.linalg.norm(src_centered, axis=1))
    dst_norm_factor = np.mean(np.linalg.norm(dst_centered, axis=1))

    if src_norm_factor < 1e-6 or dst_norm_factor < 1e-6:
        print("警告：Sim3 RANSAC 输入点集分布非常集中，归一化可能不稳定。")
        src_normalized = src_centered
        dst_normalized = dst_centered
    else:
        src_normalized = src_centered / src_norm_factor
        dst_normalized = dst_centered / dst_norm_factor

    try:
        ransac_x = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)
        ransac_y = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)
        ransac_z = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)

        ransac_x.fit(src_normalized, dst_normalized[:, 0])
        ransac_y.fit(src_normalized, dst_normalized[:, 1])
        ransac_z.fit(src_normalized, dst_normalized[:, 2])

        inlier_mask = ransac_x.inlier_mask_ & ransac_y.inlier_mask_ & ransac_z.inlier_mask_
        num_inliers = np.sum(inlier_mask)
        print(f"Sim3 RANSAC: 找到 {num_inliers} / {src.shape[0]} 个内点")

        if num_inliers < min_inliers_needed:
            raise ValueError(f"Sim3 RANSAC: 有效内点不足 ({num_inliers} < {min_inliers_needed})，无法计算可靠的 Sim3 变换")

        return compute_sim3_transform(src[inlier_mask], dst[inlier_mask])

    except ValueError as ve:
         print(f"Sim3 RANSAC 失败: {ve}. 无法继续。")
         raise ve
    except Exception as e:
         print(f"Sim3 RANSAC 过程失败: {e}.")
         if src.shape[0] < 3:
              raise ValueError("点数不足 (<3)，无法计算 Sim3 变换")
         raise ValueError(f"Sim3 RANSAC 失败 ({e}) 且无法使用所有点计算。")


def compute_sim3_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """计算从源点(src)到目标点(dst)的最佳 Sim3 变换 (旋转R, 平移t, 尺度s)"""
    if src.shape[0] < 3:
         raise ValueError(f"计算 Sim3 变换需要至少 3 个点，但只有 {src.shape[0]} 个")
    if src.shape != dst.shape:
         raise ValueError("Sim3 计算：源点和目标点数量必须一致")

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
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    src_dist = np.linalg.norm(src_centered, axis=1)
    dst_dist = np.linalg.norm(dst_centered, axis=1)
    valid_scale_mask = src_dist > 1e-9

    if not np.any(valid_scale_mask):
         print("警告：所有源点都靠近质心，无法可靠计算尺度，默认为 1.0")
         scale = 1.0
    else:
        ratios = dst_dist[valid_scale_mask] / src_dist[valid_scale_mask]
        scale = np.median(ratios)
        if scale <= 1e-6:
             print(f"警告：计算出的尺度非常小 ({scale:.2e})，可能存在问题。重置为 1.0")
             scale = 1.0

    t = dst_centroid - scale * (R @ src_centroid)
    print(f"计算得到的 Sim3 参数: scale={scale:.4f}")
    return R, t, scale

# ----------------------------
# 轨迹变换与姿态更新
# ----------------------------

def transform_trajectory(positions: np.ndarray, quaternions: np.ndarray,
                        R: np.ndarray, t: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """应用Sim3变换到整个轨迹（位置和姿态）"""
    trans_pos = scale * (R @ positions.T).T + t

    R_sim3_rot = Rotation.from_matrix(R)
    trans_quat_list = []
    for q in quaternions:
        original_rot = Rotation.from_quat(q)
        new_rot = R_sim3_rot * original_rot
        new_quat_xyzw = new_rot.as_quat()
        trans_quat_list.append(new_quat_xyzw)

    return trans_pos, np.array(trans_quat_list)


# ----------------------------
# 可视化与评估系统
# ----------------------------

def plot_results(original_pos: np.ndarray, corrected_pos: np.ndarray,
                 gps_pos: np.ndarray, # 原始(过滤后)的GPS UTM点 (Original (filtered) GPS UTM points)
                 valid_indices_for_error: np.ndarray,
                 aligned_gps_for_error: np.ndarray, # 对齐到SLAM时间的GPS点 (GPS points aligned to SLAM time)
                 slam_times: np.ndarray # SLAM 时间戳用于绘图 (SLAM timestamps for plotting)
                ) -> None:
    """增强的可视化系统，显示原始轨迹、修正后轨迹、GPS点和误差 (Enhanced visualization system to show original trajectory, corrected trajectory, GPS points, and error)"""
    plt.figure(figsize=(15, 12))

    # --- 1. 2D轨迹对比 (XY平面) (2D Trajectory Comparison (XY Plane)) ---
    ax1 = plt.subplot(2, 2, 1)
    # Use English labels
    ax1.plot(original_pos[:, 0], original_pos[:, 1], 'b-', alpha=0.5, linewidth=1, label='Original SLAM Trajectory')
    ax1.plot(corrected_pos[:, 0], corrected_pos[:, 1], 'g-', linewidth=1.5, label='Corrected Trajectory (EKF)')
    # 绘制原始(过滤后)的GPS点作为参考 (Plot original (filtered) GPS points as reference)
    ax1.scatter(gps_pos[:, 0], gps_pos[:, 1], c='r', marker='x', s=30, label='GPS Reference Points (Filtered, UTM)')
    # 绘制用于计算误差的对齐(插值)后的GPS点 (Plot aligned (interpolated) GPS points used for error calculation)
    if len(valid_indices_for_error) > 0:
       ax1.scatter(aligned_gps_for_error[:, 0], aligned_gps_for_error[:, 1],
                   facecolors='none', edgecolors='orange', marker='o', s=40,
                   label='Aligned GPS Points (Interpolated, for Error Calc.)')

    # Use English titles and labels
    ax1.set_title('Trajectory Comparison (XY Plane)')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal') # 保持XY轴比例一致，使形状不失真 (Keep XY axis ratio consistent to avoid distortion)

    # --- 2. 3D轨迹对比 (3D Trajectory Comparison) ---
    ax3d = plt.subplot(2, 2, 2, projection='3d')
    # Use English labels
    ax3d.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 'b-', alpha=0.5, linewidth=1, label='Original SLAM Trajectory')
    ax3d.plot(corrected_pos[:, 0], corrected_pos[:, 1], corrected_pos[:, 2], 'g-', linewidth=1.5, label='Corrected Trajectory (EKF)')
    ax3d.scatter(gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2], c='r', marker='x', s=30, label='GPS Reference Points (Filtered, UTM)')
    if len(valid_indices_for_error) > 0:
        ax3d.scatter(aligned_gps_for_error[:, 0], aligned_gps_for_error[:, 1], aligned_gps_for_error[:, 2],
                    facecolors='none', edgecolors='orange', marker='o', s=40,
                    label='Aligned GPS Points (Interpolated, for Error Calc.)')

    # Use English titles and labels
    ax3d.set_title('Trajectory Comparison (3D)')
    ax3d.set_xlabel('X (meters)')
    ax3d.set_ylabel('Y (meters)')
    ax3d.set_zlabel('Z (meters)')
    ax3d.legend()
    # 尝试自动调整3D视图范围以包含主要轨迹 (Attempt to auto-adjust 3D view range to include main trajectory)
    try:
        max_range = np.array([corrected_pos[:,0].max()-corrected_pos[:,0].min(),
                              corrected_pos[:,1].max()-corrected_pos[:,1].min(),
                              corrected_pos[:,2].max()-corrected_pos[:,2].min()]).max() / 2.0
        if max_range < 1.0: max_range = 1.0 # 避免范围过小 (Avoid too small range)
        mid_x = (corrected_pos[:,0].max()+corrected_pos[:,0].min()) * 0.5
        mid_y = (corrected_pos[:,1].max()+corrected_pos[:,1].min()) * 0.5
        mid_z = (corrected_pos[:,2].max()+corrected_pos[:,2].min()) * 0.5
        ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    except: # 如果计算失败（例如轨迹点很少）(If calculation fails (e.g., too few trajectory points))
        pass # 使用默认视图 (Use default view)

    # --- 3. 最终误差分布 (Final Error Distribution) ---
    ax_hist = plt.subplot(2, 2, 3)
    if len(valid_indices_for_error) > 0:
        # 计算最终误差 (EKF结果 vs 对齐后的GPS) (Calculate final error (EKF result vs Aligned GPS))
        final_errors = np.linalg.norm(corrected_pos[valid_indices_for_error] - aligned_gps_for_error, axis=1)

        ax_hist.hist(final_errors, bins=30, alpha=0.7, color='purple')
        mean_err = np.mean(final_errors)
        std_err = np.std(final_errors)
        max_err = np.max(final_errors)
        # Use English title and labels
        ax_hist.set_title(f'Final Position Error Distribution\nMean: {mean_err:.2f}m, StdDev: {std_err:.2f}m, Max: {max_err:.2f}m')
        ax_hist.set_xlabel('Error (meters)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.grid(True)
    else:
        # Use English title and text
        ax_hist.set_title("Cannot Calculate Error: No Valid Aligned Points")
        ax_hist.text(0.5, 0.5, "No Error Data", ha='center', va='center')

    # --- 4. 误差随时间变化图 (Error Over Time Plot) ---
    ax_err_time = plt.subplot(2, 2, 4)
    if len(valid_indices_for_error) > 0 and len(slam_times) == len(corrected_pos):
        # 使用有效索引对应的时间戳 (Use timestamps corresponding to valid indices)
        valid_timestamps = slam_times[valid_indices_for_error]
        if len(valid_timestamps) == len(final_errors):
             # 计算相对时间 (从第一个有效点开始) (Calculate relative time (from the first valid point))
             relative_time = valid_timestamps - valid_timestamps[0]
             # Use English labels and title
             ax_err_time.plot(relative_time, final_errors, 'r-', linewidth=1, label='Absolute Position Error')
             ax_err_time.set_xlabel('Relative Time (seconds)')
             ax_err_time.set_ylabel('Error (meters)')
             ax_err_time.set_title('Error Over Time')
             ax_err_time.grid(True)
             ax_err_time.legend()
        else:
             # 如果时间戳和误差长度不匹配（理论上不应发生），绘制散点图作为后备
             # (If timestamps and error length don't match (shouldn't happen theoretically), plot scatter as fallback)
             # Use English labels and title
             scatter_plot = ax_err_time.scatter(corrected_pos[valid_indices_for_error, 0],
                                                corrected_pos[valid_indices_for_error, 1],
                                                c=final_errors, cmap='viridis', s=10, label='Abs. Pos. Error (m)')
             plt.colorbar(scatter_plot, ax=ax_err_time)
             ax_err_time.set_title('Error Spatial Distribution (XY Plane)')
             ax_err_time.set_xlabel('X (meters)')
             ax_err_time.set_ylabel('Y (meters)')
             ax_err_time.axis('equal')
             ax_err_time.legend()

    else:
         # Use English title and text
         ax_err_time.set_title("Cannot Plot Error: No Valid Points or Timestamps")
         ax_err_time.text(0.5, 0.5, "No Error Data", ha='center', va='center')

    plt.tight_layout() # 自动调整子图间距 (Automatically adjust subplot spacing)
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
        过程噪声 (Q): 7x7, 代表模型预测的不确定性
        测量噪声 (R): 3x3, 代表GPS测量的不确定性 (只测位置)
        """
        initial_quat_normalized = self.normalize_quaternion(initial_quat)
        self.state = np.concatenate([initial_pos, initial_quat_normalized]).astype(float)
        self.cov = np.diag(initial_cov_diag).astype(float)
        self.Q = np.diag(process_noise_diag).astype(float)
        self.R = np.diag(meas_noise_diag).astype(float)

        if self.state.shape != (7,): raise ValueError(f"EKF: 初始状态维度错误 (应7, 实际{self.state.shape})")
        if self.cov.shape != (7, 7): raise ValueError(f"EKF: 初始协方差维度错误 (应7x7, 实际{self.cov.shape})")
        if self.Q.shape != (7, 7): raise ValueError(f"EKF: 过程噪声Q维度错误 (应7x7, 实际{self.Q.shape})")
        if self.R.shape != (3, 3): raise ValueError(f"EKF: 测量噪声R维度错误 (应3x3, 实际{self.R.shape})")

    def predict(self, slam_pos: np.ndarray, slam_quat: np.ndarray, delta_time: float) -> None:
        """预测步骤：使用SLAM位姿作为预测值，增加不确定性"""
        self.state[:3] = slam_pos
        self.state[3:] = self.normalize_quaternion(slam_quat)

        # P = P + Q*dt (简化模型)
        dt_adjusted = max(delta_time, 1e-6)
        self.cov += self.Q * dt_adjusted
        self.cov = (self.cov + self.cov.T) / 2.0 # 保持对称

    def update(self, gps_pos: np.ndarray) -> bool:
        """更新步骤：使用 GPS 测量值修正状态估计"""
        if gps_pos.shape != (3,):
            # print(f"警告: EKF 更新收到的 GPS 位置维度错误 (应为 3)，跳过更新。") # 减少冗余打印
            return False
        if np.isnan(gps_pos).any():
             # print(f"警告: EKF 更新收到的 GPS 位置包含 NaN，跳过更新。") # 减少冗余打印
             return False

        # 测量矩阵 H
        H = np.zeros((3, 7))
        H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 2] = 1.0

        try:
            # 测量残差 y = z - H*x_pred
            innovation = gps_pos - self.state[:3]

            # 残差协方差 S = H*P*H^T + R
            H_P = H @ self.cov
            S = H_P @ H.T + self.R

            # 检查 S 是否奇异
            det_S = np.linalg.det(S)
            if abs(det_S) < 1e-12:
                 # print(f"警告：EKF 更新步骤中 S 矩阵接近奇异 (det={det_S:.2e})，跳过本次更新。") # 减少冗余打印
                 return False

            # 卡尔曼增益 K = P*H^T*S^(-1)
            S_inv = np.linalg.inv(S)
            K = self.cov @ H.T @ S_inv

            # 状态更新 x = x + K*y
            self.state += K @ innovation
            self.state[3:] = self.normalize_quaternion(self.state[3:]) # 归一化四元数

            # 协方差更新 P = (I - K*H)*P
            I = np.eye(7)
            self.cov = (I - K @ H) @ self.cov
            self.cov = (self.cov + self.cov.T) / 2.0 # 保持对称

            return True # 更新成功

        except np.linalg.LinAlgError as e:
            print(f"警告：EKF 更新步骤中发生线性代数错误: {e}。跳过本次更新。")
            return False
        except Exception as e:
            print(f"警告：EKF 更新步骤中发生未知错误: {e}。跳过本次更新。")
            traceback.print_exc()
            return False

    @staticmethod
    def normalize_quaternion(q: np.ndarray) -> np.ndarray:
        """归一化四元数"""
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            print("警告：尝试归一化零四元数，返回默认单位四元数 [0,0,0,1]。")
            return np.array([0.0, 0.0, 0.0, 1.0])
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
        return np.empty((0, 3)), np.empty((0, 4))

    ekf_config = config['ekf'] # 获取 EKF 配置
    time_align_config = config['time_alignment'] # 获取时间对齐配置

    # 1. 初始化 EKF
    try:
        ekf = ExtendedKalmanFilter(
            initial_pos=sim3_pos[0],
            initial_quat=sim3_quat[0],
            initial_cov_diag=ekf_config['initial_cov_diag'],
            process_noise_diag=ekf_config['process_noise_diag'],
            meas_noise_diag=ekf_config['meas_noise_diag']
        )
    except Exception as e:
         raise ValueError(f"EKF 初始化失败: {e}")

    # 2. 获取对齐的 GPS 数据用于更新 (包含中断处理)
    #    使用修改后的 dynamic_time_alignment 函数
    print("正在执行分段时间对齐以获取 EKF 更新...")
    aligned_gps, valid_mask = dynamic_time_alignment(
        slam_data, gps_data, time_align_config # 传递时间对齐配置
    )
    # 获取那些有有效对齐 GPS 数据的 SLAM 时间戳的索引
    valid_indices_for_update = np.where(valid_mask)[0]

    print(f"EKF 修正：将使用 {len(valid_indices_for_update)} 个有效的 GPS 测量进行更新 (已跳过中断部分)。")

    # 准备存储修正后的轨迹
    corrected_pos_list = [ekf.state[:3].copy()]
    corrected_quat_list = [ekf.state[3:].copy()]
    last_time = slam_data['timestamps'][0]
    updates_skipped = 0
    updates_applied = 0

    # 3. 迭代处理每个 SLAM 时间点
    for i in range(1, n_points):
        current_time = slam_data['timestamps'][i]
        delta_t = current_time - last_time

        # a) EKF 预测步骤
        ekf.predict(sim3_pos[i], sim3_quat[i], delta_t)

        # b) EKF 更新步骤：只在当前时间点有 *有效* (非 NaN) 对齐 GPS 时执行
        if valid_mask[i]: # 直接检查 valid_mask
            gps_measurement = aligned_gps[i]
            # 不需要再次检查 NaN，因为 valid_mask 已经保证了这一点
            update_successful = ekf.update(gps_measurement)
            if update_successful:
                 updates_applied += 1
            else:
                 updates_skipped += 1 # 更新失败（如矩阵奇异）
        # else: # 如果 valid_mask[i] 是 False (即处于中断或插值无效区域)，则不执行更新

        # 记录当前 EKF 修正后的状态
        corrected_pos_list.append(ekf.state[:3].copy())
        corrected_quat_list.append(ekf.state[3:].copy())
        last_time = current_time

    print(f"EKF 修正完成。在 {n_points-1} 步预测中，成功应用了 {updates_applied} 次 GPS 更新，因内部问题跳过 {updates_skipped} 次更新。")
    print(f"共有 {n_points - 1 - np.sum(valid_mask[1:])} 步仅执行了 SLAM 预测 (无有效 GPS 更新)。")


    # 确保列表长度与输入一致
    if len(corrected_pos_list) != n_points:
         print(f"严重警告：EKF 输出长度 ({len(corrected_pos_list)}) 与输入长度 ({n_points}) 不匹配！可能导致后续错误。")
         # 尝试修正（例如重复最后一个有效点），但这只是权宜之计
         while len(corrected_pos_list) < n_points:
             corrected_pos_list.append(corrected_pos_list[-1])
             corrected_quat_list.append(corrected_quat_list[-1])
         if len(corrected_pos_list) > n_points:
             corrected_pos_list = corrected_pos_list[:n_points]
             corrected_quat_list = corrected_quat_list[:n_points]

    return np.array(corrected_pos_list), np.array(corrected_quat_list)


# ----------------------------
# 主流程控制
# ----------------------------

def main_process_gui():
    """主处理流程，包含GUI文件选择、处理和可视化"""
    try:
        # 1. 文件选择
        slam_path = select_slam_file()
        if not slam_path: print("未选择 SLAM 文件，操作取消。"); return
        gps_path = select_gps_file()
        if not gps_path: print("未选择 GPS 文件，操作取消。"); return

        print(f"选择的 SLAM 文件: {slam_path}")
        print(f"选择的 GPS 文件: {gps_path}")

        # 2. 数据加载 (包含 GPS RANSAC 过滤)
        print("正在加载数据...")
        slam_data = load_slam_trajectory(slam_path)
        gps_data = load_gps_data(gps_path) # gps_data['positions'] 已是过滤后的 UTM
        print(f"SLAM 点数: {len(slam_data['positions'])}")
        print(f"GPS 点数 (有效、投影并过滤后): {len(gps_data['positions'])}")
        if len(slam_data['positions']) == 0 or len(gps_data['positions']) < 2:
             raise ValueError("SLAM 数据为空 或 有效 GPS 数据点不足 (<2)，无法继续处理。")

        # 3. 时间对齐 (获取用于 Sim3 的匹配点 - 使用分段对齐)
        print("正在进行时间对齐以获取 Sim3 匹配点 (支持中断处理)...")
        # 注意：这里也使用修改后的对齐函数，因为它返回的结果用于Sim3
        aligned_gps_for_sim3, valid_mask_sim3 = dynamic_time_alignment(
            slam_data, gps_data, CONFIG['time_alignment'] # 传递时间对齐配置
        )
        valid_indices_sim3 = np.where(valid_mask_sim3)[0] # 只使用有效对齐的点进行Sim3

        # 检查是否有足够的匹配点进行 Sim3 RANSAC
        min_points_needed = CONFIG['sim3_ransac']['min_inliers_for_transform']
        if len(valid_indices_sim3) < min_points_needed:
            raise ValueError(f"有效时间同步匹配点不足 ({len(valid_indices_sim3)} < {min_points_needed})，无法进行 Sim3 变换估计。请检查时间戳、中断阈值或数据质量。")
        print(f"找到 {len(valid_indices_sim3)} 个有效时间同步的匹配点用于 Sim3 估计。")

        # 4. Sim3 全局对齐 (使用 RANSAC)
        print("正在计算稳健的 Sim3 全局变换...")
        src_points = slam_data['positions'][valid_indices_sim3]
        dst_points = aligned_gps_for_sim3[valid_indices_sim3] # 确保只用有效对齐的点

        R, t, scale = compute_sim3_transform_robust(
            src=src_points,
            dst=dst_points,
            min_samples=CONFIG['sim3_ransac']['min_samples'],
            residual_threshold=CONFIG['sim3_ransac']['residual_threshold'],
            max_trials=CONFIG['sim3_ransac']['max_trials'],
            min_inliers_needed=CONFIG['sim3_ransac']['min_inliers_for_transform']
        )

        print("正在应用 Sim3 变换到整个 SLAM 轨迹...")
        sim3_pos, sim3_quat = transform_trajectory(
            slam_data['positions'],
            slam_data['quaternions'],
            R, t, scale
        )

        # 5. EKF 局部修正 (融合)
        print("正在应用 EKF 进行轨迹融合与修正 (支持中断处理)...")
        # 将 Sim3 变换后的轨迹 和 原始(过滤后)的 GPS 数据 送入 EKF
        # apply_ekf_correction 内部会调用修改后的 dynamic_time_alignment
        corrected_pos, corrected_quat = apply_ekf_correction(
            slam_data,
            gps_data,
            sim3_pos,
            sim3_quat,
            CONFIG # 传递完整的配置字典
        )

        # 6. 评估误差 (使用最终修正后的轨迹与对齐的GPS点进行比较)
        print("正在评估轨迹误差...")
        # 需要再次获取对齐的 GPS 点（插值到 SLAM 时间戳上）用于评估
        # 注意：这里使用的 aligned_gps_for_eval 应该与 EKF 更新时内部使用的对齐结果一致
        # *** 同样需要传递时间对齐配置 ***

        # CORRECT: Calculate aligned GPS specifically for evaluation
        aligned_gps_for_eval, valid_mask_eval = dynamic_time_alignment(
             slam_data, gps_data, CONFIG['time_alignment'] # <-- 传递配置
        )
        # Get the indices where alignment was successful (and not during an interruption)
        valid_indices_eval = np.where(valid_mask_eval)[0]

        eval_gps_points = np.empty((0,3)) # Initialize an empty array for the points used in evaluation
        if len(valid_indices_eval) > 0:
            # CORRECT: Use the *calculated* aligned_gps_for_eval to get the actual points
            eval_gps_points = aligned_gps_for_eval[valid_indices_eval] # Extract the valid aligned GPS points

            # --- Now use eval_gps_points for comparisons ---
            # a) 原始 SLAM 轨迹 vs 对齐的 GPS (评估初始误差)
            raw_errors = np.linalg.norm(slam_data['positions'][valid_indices_eval] - eval_gps_points, axis=1)
            print(f"[评估] 原始轨迹   vs 对齐GPS -> 均值误差: {np.mean(raw_errors):.3f} m, 标准差: {np.std(raw_errors):.3f} m, 最大误差: {np.max(raw_errors):.3f} m (基于 {len(valid_indices_eval)} 个有效点)")

            # b) Sim3 对齐后 vs 对齐的 GPS (评估全局对齐效果)
            sim3_errors = np.linalg.norm(sim3_pos[valid_indices_eval] - eval_gps_points, axis=1)
            print(f"[评估] Sim3 对齐后 vs 对齐GPS -> 均值误差: {np.mean(sim3_errors):.3f} m, 标准差: {np.std(sim3_errors):.3f} m, 最大误差: {np.max(sim3_errors):.3f} m (基于 {len(valid_indices_eval)} 个有效点)")

            # c) EKF 融合后 vs 对齐的 GPS (评估最终效果)
            ekf_errors = np.linalg.norm(corrected_pos[valid_indices_eval] - eval_gps_points, axis=1)
            print(f"[评估] EKF 融合后  vs 对齐GPS -> 均值误差: {np.mean(ekf_errors):.3f} m, 标准差: {np.std(ekf_errors):.3f} m, 最大误差: {np.max(ekf_errors):.3f} m (基于 {len(valid_indices_eval)} 个有效点)")
        else:
            print("警告：无有效对齐的GPS点用于最终误差评估 (可能全程中断或时间不重叠)。")
            ekf_errors = np.array([]) # Used for the plotting function



        # 7. 保存结果
        save_results = messagebox.askyesno("保存结果", "处理完成。是否要保存修正后的轨迹?")
        if save_results:
            output_path_utm = filedialog.asksaveasfilename(
                title="保存修正后的轨迹 (UTM 坐标, TUM 格式)",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt")],
                initialfile="corrected_trajectory_utm.txt"
            )
            if output_path_utm:
                output_data_utm = np.column_stack((
                    slam_data['timestamps'],
                    corrected_pos,
                    corrected_quat
                ))
                np.savetxt(output_path_utm, output_data_utm,
                           fmt='%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f',
                           header="timestamp x y z qx qy qz qw (UTM)",
                           comments='')
                print(f"UTM 轨迹已保存至: {output_path_utm}")

                try:
                    projector = gps_data.get('projector')
                    if projector:
                        print("正在将修正后的轨迹转换回 WGS84 坐标...")
                        wgs84_lon_lat_alt = utm_to_wgs84(corrected_pos, projector)
                        output_data_wgs84 = np.column_stack((
                            slam_data['timestamps'],
                            wgs84_lon_lat_alt, # [lon, lat, alt]
                            corrected_quat
                        ))
                        output_path_wgs84 = output_path_utm.replace('_utm.txt', '_wgs84.txt')
                        if output_path_wgs84 == output_path_utm:
                             output_path_wgs84 = output_path_utm.replace('.txt', '_wgs84.txt')

                        np.savetxt(output_path_wgs84, output_data_wgs84,
                                   fmt='%.6f %.8f %.8f %.3f %.6f %.6f %.6f %.6f', # lon/lat更高精度
                                   header="timestamp lon lat alt qx qy qz qw (WGS84)",
                                   comments='')
                        print(f"WGS84 轨迹已保存至: {output_path_wgs84}")
                    else:
                         print("警告：GPS 数据中缺少投影仪对象 'projector'，无法保存 WGS84 轨迹。")
                except Exception as e:
                    print(f"保存 WGS84 轨迹失败: {str(e)}")
                    traceback.print_exc()

        # 8. 可视化 (传递 ekf_errors 给绘图函数)
        print("正在生成可视化结果...")
        # 确保将 ekf_errors (可能为空) 传递给绘图函数
        # 同时确保 valid_indices_eval 和 eval_gps_points 也是对应的
        plot_results(
            original_pos=slam_data['positions'],
            corrected_pos=corrected_pos,
            gps_pos=gps_data['positions'], # 过滤后的原始GPS点
            valid_indices_for_error=valid_indices_eval, # EKF有效更新点的索引
            aligned_gps_for_error=eval_gps_points,      # EKF有效更新点对应的插值GPS位置
            slam_times=slam_data['timestamps']
        )

        print("处理完成。")
        messagebox.showinfo("完成", "轨迹对齐与融合处理完成！")

    except ValueError as ve:
        error_msg = f"处理失败 (值错误): {str(ve)}"
        print(error_msg); traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except FileNotFoundError as fnf:
        error_msg = f"处理失败 (文件未找到): {str(fnf)}"
        print(error_msg); messagebox.showerror("处理失败", error_msg)
    except AssertionError as ae:
        error_msg = f"处理失败 (数据格式断言错误): {str(ae)}"
        print(error_msg); traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except CRSError as ce:
         error_msg = f"处理失败 (坐标投影错误): {str(ce)}"
         print(error_msg); traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except np.linalg.LinAlgError as lae:
         error_msg = f"处理失败 (线性代数计算错误): {str(lae)}"
         print(error_msg); traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except Exception as e:
        error_msg = f"处理过程中发生未预料的错误: {str(e)}"
        print(error_msg); traceback.print_exc()
        messagebox.showerror("处理失败", f"{error_msg}\n\n详情请查看控制台输出。")

if __name__ == "__main__":
    # 设置 matplotlib 支持中文显示 (如果需要，取消注释并确保安装了相应字体)
    # plt.rcParams['font.sans-serif'] = ['SimHei'] # 或者其他支持中文的字体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    print("启动 SLAM-GPS 轨迹对齐与融合工具 (支持GPS中断处理)...")
    main_process_gui()
    print("程序结束。")
