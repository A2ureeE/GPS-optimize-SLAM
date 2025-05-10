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
# 配置参数
# ----------------------------
CONFIG = {
    # EKF 参数
    "ekf": {
        "initial_cov_diag": [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01], # 初始协方差矩阵对角线元素 (x,y,z, qx,qy,qz,qw)
        "process_noise_diag": [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01], # 过程噪声协方差矩阵对角线元素 (每秒，近似值，对应 x,y,z, qx,qy,qz,qw)
        "meas_noise_diag": [0.3, 0.3, 0.3], # GPS x,y,z 测量噪声标准差
        "transition_steps": 15,         # GNSS 恢复时，平滑过渡所需的步数
    },
    # Sim(3) 全局变换 RANSAC 参数
    "sim3_ransac": {
        "min_samples": 4, # RANSAC 最小样本数
        "residual_threshold": 0.1, # RANSAC 残差阈值
        "max_trials": 1000, # RANSAC 最大尝试次数
        "min_inliers_needed": 4, # 计算Sim3变换所需的最少内点数
    },
    # GPS 轨迹 RANSAC 滤波参数
    "gps_filtering_ransac": {
        "enabled": True, # 是否启用GPS RANSAC滤波
        "use_sliding_window": True, # 是否使用滑动窗口进行RANSAC滤波
        "window_duration_seconds": 15.0, # 滑动窗口时长 (秒)
        "window_step_factor": 0.5, # 滑动窗口步长因子 (相对于窗口时长)
        "polynomial_degree": 2, # RANSAC拟合的多项式阶数
        "min_samples": 6, # RANSAC每个窗口或全局拟合的最小样本数
        "residual_threshold_meters": 10.0, # RANSAC残差阈值 (米)
        "max_trials": 50, # RANSAC最大尝试次数
    },
    # 时间对齐参数
    "time_alignment": {
        "max_samples_for_corr": 500, # 用于互相关估计时间偏移的最大样本数
        "max_gps_gap_threshold": 5.0, # GPS数据中允许的最大时间间隙 (秒)，超过则认为中断
    }
}

# ----------------------------
# 辅助函数
# ----------------------------

def calculate_relative_pose(pose1_pos: np.ndarray, pose1_quat: np.ndarray,
                            pose2_pos: np.ndarray, pose2_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算从 pose1 到 pose2 的相对运动。

    输入:
        pose1_pos (np.ndarray): 起始位置 [x, y, z]
        pose1_quat (np.ndarray): 起始姿态四元数 [x, y, z, w] (scipy格式)
        pose2_pos (np.ndarray): 结束位置 [x, y, z]
        pose2_quat (np.ndarray): 结束姿态四元数 [x, y, z, w] (scipy格式)

    输出:
        Tuple[np.ndarray, np.ndarray]:
            - delta_pos_local (np.ndarray): 在 pose1 坐标系下的位置变化 [dx, dy, dz]
            - delta_quat (np.ndarray): 相对旋转四元数 [x, y, z, w] (从 pose1 到 pose2)
    """
    try:
        rot1 = Rotation.from_quat(pose1_quat) # 从四元数创建旋转对象 pose1
        rot1_inv = rot1.inv() # 计算 pose1 旋转的逆
        rot2 = Rotation.from_quat(pose2_quat) # 从四元数创建旋转对象 pose2
    except ValueError as e: # 捕获无效四元数导致的错误
        print(f"警告 (calculate_relative_pose): 无效的四元数输入: {e}. 返回零运动。") # 打印警告信息
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]) # 返回零平移和单位旋转

    # 计算在世界坐标系下的位置差
    delta_pos_world = pose2_pos - pose1_pos # 计算世界坐标系下的位置差

    # 将世界坐标系下的位置差转换到 pose1 的局部坐标系下
    delta_pos_local = rot1_inv.apply(delta_pos_world) # 将世界坐标系位置差转换到 pose1 的局部坐标系

    # 计算相对旋转：delta_rot = rot1_inv * rot2
    delta_rot = rot1_inv * rot2 # 计算从 pose1 到 pose2 的相对旋转
    delta_quat = delta_rot.as_quat() # 将相对旋转转换为四元数

    return delta_pos_local, delta_quat # 返回局部坐标系下的位置变化和相对旋转四元数

def quaternion_nlerp(q1: np.ndarray, q2: np.ndarray, weight_q2: float) -> np.ndarray:
    """
    对两个四元数进行归一化线性插值 (NLERP)。

    输入:
        q1 (np.ndarray): 第一个四元数 [x, y, z, w]
        q2 (np.ndarray): 第二个四元数 [x, y, z, w]
        weight_q2 (float): q2 的权重 (范围 0 到 1)。当 weight_q2=0 时结果为 q1, weight_q2=1 时结果为 q2。

    输出:
        np.ndarray: 插值并归一化后的四元数 [x, y, z, w]
    """
    # 处理方向：确保点积为正，插值走最短路径
    dot = np.dot(q1, q2) # 计算两个四元数的点积
    if dot < 0.0: # 如果点积为负，说明两个四元数代表的旋转方向相反（超过180度）
        q2 = -q2 # 反转 q2 以选择最短路径
        dot = -dot # 更新点积

    # 线性插值
    w = np.clip(weight_q2, 0.0, 1.0) # 将权重限制在 [0, 1] 范围内，确保安全
    q_interp = (1.0 - w) * q1 + w * q2 # 执行线性插值

    # 归一化结果
    norm = np.linalg.norm(q_interp) # 计算插值后四元数的模长
    if norm < 1e-9: # 如果模长过小（接近零）
        # 如果插值结果接近零（例如权重在边界且输入相同但符号相反），返回其中一个输入
        return q1 if weight_q2 < 0.5 else q2 # 根据权重返回 q1 或 q2
    return q_interp / norm # 返回归一化后的插值四元数

# ----------------------------
# 增强数据加载与预处理
# ----------------------------

def load_slam_trajectory(txt_path: str) -> Dict[str, np.ndarray]:
    """
    加载并验证SLAM轨迹数据 (TUM格式)。

    输入:
        txt_path (str): SLAM轨迹文件的路径 (TUM格式: timestamp tx ty tz qx qy qz qw)

    输出:
        Dict[str, np.ndarray]: 包含 'timestamps', 'positions', 'quaternions' 的字典。
                                'timestamps': 时间戳数组 (N,)
                                'positions': 位置数组 (N, 3) [x, y, z]
                                'quaternions': 四元数数组 (N, 4) [x, y, z, w]
    """
    try:
        data = np.loadtxt(txt_path) # 从文本文件加载数据
        if data.ndim == 1: # 如果数据只有一行
             data = data.reshape(1, -1) # 将其转换为二维数组
        assert data.shape[1] == 8, f"SLAM文件格式错误：需要8列 (ts x y z qx qy qz qw), 找到 {data.shape[1]} 列" # 断言数据列数为8
        return {
            'timestamps': data[:, 0].astype(float), # 提取时间戳并转换为浮点数
            'positions': data[:, 1:4].astype(float), # 提取位置数据并转换为浮点数
            'quaternions': data[:, 4:8].astype(float) # 提取四元数数据并转换为浮点数 (Scipy使用 [x, y, z, w] 格式)
        }
    except FileNotFoundError: # 捕获文件未找到错误
        raise ValueError(f"SLAM文件未找到: {txt_path}") # 抛出值错误
    except Exception as e: # 捕获其他加载或解析错误
        raise ValueError(f"SLAM数据加载或解析失败 ({txt_path}): {str(e)}") # 抛出值错误

def auto_utm_projection(lons: np.ndarray, lats: np.ndarray) -> Tuple[int, str]:
    """
    根据经度自动计算UTM分区号和南北半球。

    输入:
        lons (np.ndarray): 经度数组 (度)
        lats (np.ndarray): 纬度数组 (度)

    输出:
        Tuple[int, str]:
            - zone (int): UTM 分区号 (例如 50)
            - hemisphere (str): 南北半球指示字符串 (例如 '' 表示北半球, ' +south' 表示南半球)
    """
    if lons.size == 0 or lats.size == 0: # 检查经纬度数据是否为空
        raise ValueError("经纬度数据不能为空以确定UTM分区") # 如果为空则抛出值错误
    central_lon = np.mean(lons) # 计算平均经度作为中央经线
    zone = int((central_lon + 180) // 6 + 1) # 根据中央经线计算UTM分区号
    hemisphere = ' +south' if np.mean(lats) < 0 else '' # 根据平均纬度判断南北半球
    return zone, hemisphere # 返回UTM分区号和半球指示

# -------------------------------------------
# GPS 离群点过滤模块
# -------------------------------------------
def filter_gps_outliers_ransac(times: np.ndarray, positions: np.ndarray,
                               config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 RANSAC 和多项式模型过滤 GPS 轨迹中的离群点。
    支持全局拟合或滑动窗口局部拟合。

    输入:
        times (np.ndarray): GPS 时间戳数组 (N,)
        positions (np.ndarray): GPS 位置数组 (N, 3) [x, y, z], 通常是UTM坐标
        config (Dict[str, Any]): GPS RANSAC 过滤的配置参数字典，通常来自 CONFIG['gps_filtering_ransac']

    输出:
        Tuple[np.ndarray, np.ndarray]:
            - filtered_times (np.ndarray): 过滤后的时间戳数组
            - filtered_positions (np.ndarray): 过滤后的位置数组
    """
    if not config.get("enabled", False): # 检查配置中是否启用了RANSAC过滤
        print("GPS RANSAC 过滤被禁用。") # 如果未启用，打印信息
        return times, positions # 返回原始时间和位置数据

    n_points = len(times) # 获取GPS点的数量
    min_samples_needed = config['min_samples'] # 获取RANSAC所需的最小样本数（全局或每个窗口）

    if n_points < min_samples_needed: # 如果GPS点数少于RANSAC最小样本数
        print(f"警告: GPS 点数 ({n_points}) 少于 RANSAC 最小样本数 ({min_samples_needed})，跳过 GPS 离群点过滤。") # 打印警告信息
        return times, positions # 返回原始时间和位置数据

    use_sliding_window = config.get("use_sliding_window", False) # 检查是否使用滑动窗口

    if not use_sliding_window: # 如果不使用滑动窗口，执行全局RANSAC拟合
        # --- 执行原始的全局 RANSAC 拟合 ---
        print("正在执行全局 GPS RANSAC 过滤...") # 打印信息
        try:
            t_feature = times.reshape(-1, 1) # 将时间戳转换为特征形式 (N, 1)
            inlier_masks = [] # 初始化内点掩码列表
            for i in range(positions.shape[1]): # 对 X, Y, Z 坐标分别进行拟合
                target = positions[:, i] # 获取当前维度的数据
                model = make_pipeline( # 创建一个处理管道
                    PolynomialFeatures(degree=config['polynomial_degree']), # 使用多项式特征
                    RANSACRegressor( # 使用RANSAC回归器
                        min_samples=min_samples_needed, # 设置最小样本数
                        residual_threshold=config['residual_threshold_meters'], # 设置残差阈值
                        max_trials=config['max_trials'], # 设置最大尝试次数
                    )
                )
                model.fit(t_feature, target) # 拟合模型
                inlier_mask_dim = model[-1].inlier_mask_ # 获取当前维度的内点掩码
                inlier_masks.append(inlier_mask_dim) # 将掩码添加到列表

            final_inlier_mask = np.logical_and.reduce(inlier_masks) # 要求 X,Y,Z 都是内点，对各维度掩码取逻辑与
            num_inliers = np.sum(final_inlier_mask) # 计算内点数量
            num_outliers = n_points - num_inliers # 计算离群点数量

            if num_outliers > 0: # 如果存在离群点
                print(f"  全局 RANSAC: 识别并移除了 {num_outliers} 个离群点 (保留 {num_inliers} / {n_points} 个点)。") # 打印移除信息
            else: # 如果未发现离群点
                print("  全局 RANSAC: 未发现离群点。") # 打印信息

            if num_inliers < min_samples_needed: # 如果过滤后剩余点数过少
                 print(f"警告: 全局 RANSAC 过滤后剩余的 GPS 点数 ({num_inliers}) 过少。") # 打印警告

            return times[final_inlier_mask], positions[final_inlier_mask] # 返回过滤后的时间和位置数据

        except Exception as e: # 捕获全局RANSAC过程中的错误
            print(f"全局 GPS RANSAC 过滤过程中发生错误: {e}. 跳过过滤步骤。") # 打印错误信息
            traceback.print_exc() # 打印详细的错误追踪信息
            return times, positions # 返回原始时间和位置数据
        # --- 全局 RANSAC 结束 ---

    else: # 如果使用滑动窗口，执行滑动窗口RANSAC拟合
        # --- 执行滑动窗口 RANSAC 拟合 ---
        window_duration = config['window_duration_seconds'] # 获取窗口时长
        step_factor = config['window_step_factor'] # 获取窗口步长因子
        window_step = window_duration * step_factor # 计算窗口步长
        residual_threshold = config['residual_threshold_meters'] # 获取残差阈值
        poly_degree = config['polynomial_degree'] # 获取多项式阶数
        max_trials_per_window = config['max_trials'] # 获取每个窗口的最大尝试次数

        print(f"正在执行滑动窗口 GPS RANSAC 过滤 (窗口时长: {window_duration}s, 步长: {window_step:.2f}s)...") # 打印信息

        if n_points < min_samples_needed: # 如果总点数少于窗口最小样本数
             print(f"警告：总点数 ({n_points}) 少于窗口最小样本数 ({min_samples_needed})，无法使用滑动窗口。") # 打印警告
             return times, positions # 返回原始数据，无法处理

        overall_inlier_mask = np.zeros(n_points, dtype=bool) # 初始化总的内点掩码，默认为全False
        processed_windows = 0 # 初始化已处理窗口计数器
        successful_windows = 0 # 初始化成功拟合窗口计数器

        start_time = times[0] # 获取轨迹的起始时间
        end_time = times[-1] # 获取轨迹的结束时间
        current_window_start = start_time # 初始化当前窗口的起始时间

        while current_window_start < end_time: # 当窗口起始时间小于轨迹结束时间时循环
            current_window_end = current_window_start + window_duration # 计算当前窗口的结束时间
            # 找到在当前窗口时间范围内的点的索引
            window_indices = np.where((times >= current_window_start) & (times < current_window_end))[0]
            n_window_points = len(window_indices) # 获取当前窗口内的点数

            if n_window_points >= min_samples_needed: # 如果窗口内的点数满足最小样本数要求
                processed_windows += 1 # 已处理窗口数加一
                window_times = times[window_indices] # 获取窗口内的时间戳
                window_positions = positions[window_indices] # 获取窗口内的位置数据
                window_t_feature = window_times.reshape(-1, 1) # 将窗口内时间戳转换为特征形式

                try:
                    window_inlier_masks_dim = [] # 初始化当前窗口各维度的内点掩码列表
                    valid_window_fit = True # 标记当前窗口拟合是否有效，默认为True
                    for i in range(positions.shape[1]): # 对 X, Y, Z 坐标分别进行拟合
                        target = window_positions[:, i] # 获取当前维度的数据
                        model = make_pipeline( # 创建处理管道
                            PolynomialFeatures(degree=poly_degree), # 使用多项式特征
                            RANSACRegressor( # 使用RANSAC回归器
                                min_samples=min_samples_needed, # 设置最小样本数
                                residual_threshold=residual_threshold, # 设置残差阈值
                                max_trials=max_trials_per_window, # 设置最大尝试次数
                            )
                        )
                        model.fit(window_t_feature, target) # 拟合模型
                        inlier_mask_window_dim = model[-1].inlier_mask_ # 获取当前维度的内点掩码
                        window_inlier_masks_dim.append(inlier_mask_window_dim) # 添加到列表

                    if valid_window_fit: # 如果窗口拟合有效
                        # 对各维度掩码取逻辑与，得到窗口最终的内点掩码
                        window_final_inlier_mask = np.logical_and.reduce(window_inlier_masks_dim)
                        # 获取这些内点在原始数据中的索引
                        original_indices_of_inliers = window_indices[window_final_inlier_mask]
                        overall_inlier_mask[original_indices_of_inliers] = True # 在总的内点掩码中标记这些点为内点
                        successful_windows += 1 # 成功拟合窗口数加一

                except Exception as e: # 捕获窗口RANSAC拟合过程中的错误
                     print(f"警告：窗口 [{current_window_start:.2f}s - {current_window_end:.2f}s] RANSAC 拟合失败: {e}") # 打印警告信息

            if window_step <= 1e-6: # 如果窗口步长非常小（或为0，避免死循环）
                # 找到下一个不同于当前窗口起始时间的时间点
                next_diff_indices = np.where(times > current_window_start)[0]
                if len(next_diff_indices) > 0: # 如果找到了
                    current_window_start = times[next_diff_indices[0]] # 更新窗口起始时间
                else: # 如果没找到（所有剩余点的时间都与当前窗口起始时间相同）
                    break # 结束循环
            else: # 如果窗口步长正常
                 current_window_start += window_step # 窗口起始时间向前滑动一个步长

            # 特殊处理：确保最后一个窗口能覆盖到轨迹的末尾数据点
            if current_window_start >= end_time and times[-1] >= current_window_end :
                 # 如果窗口起始已超过轨迹结束，且轨迹最后一个点仍在当前窗口结束时间之后（说明最后一个窗口未完全覆盖）
                 # 则将窗口起始时间调整为能包含最后一个数据点的位置
                 current_window_start = max(start_time, times[-1] - window_duration + 1e-6) # 加一个极小值避免重复处理最后一个点

        num_inliers = np.sum(overall_inlier_mask) # 计算总的内点数量
        num_outliers = n_points - num_inliers # 计算总的离群点数量

        print(f"滑动窗口 RANSAC 完成: 处理了 {processed_windows} 个窗口, 其中 {successful_windows} 个成功拟合。") # 打印处理结果
        if num_outliers > 0: # 如果存在离群点
            print(f"  滑动窗口 RANSAC: 识别并移除了 {num_outliers} 个离群点 (保留 {num_inliers} / {n_points} 个点).") # 打印移除信息
        else: # 如果未发现离群点
            print("  滑动窗口 RANSAC: 未发现离群点。") # 打印信息

        if num_inliers < 2: # 如果过滤后剩余的GPS点数过少
             print(f"警告: 滑动窗口 RANSAC 过滤后剩余的 GPS 点数 ({num_inliers}) 过少 (< 2)，可能导致后续处理失败。将返回过滤后的少量点。") # 打印警告
             # 即使点少，也返回过滤后的结果，让后续步骤决定如何处理
             return times[overall_inlier_mask], positions[overall_inlier_mask]

        return times[overall_inlier_mask], positions[overall_inlier_mask] # 返回过滤后的时间和位置数据

def load_gps_data(txt_path: str) -> Dict[str, Any]:
    """
    加载GPS数据(时间戳, 纬度, 经度, 海拔[, 其他列...]), 进行UTM投影, 并可选地过滤离群点。

    输入:
        txt_path (str): GPS数据文件的路径。文件格式应为: timestamp lat lon alt [...], 可以用空格或逗号分隔。

    输出:
        Dict[str, Any]: 包含处理后GPS数据的字典:
            'timestamps': 过滤后的时间戳数组 (N,)
            'positions': 过滤后的UTM位置数组 (N, 3) [x, y, z]
            'utm_zone': UTM分区字符串 (例如 "50N" 或 "50S")
            'projector': pyproj.Proj 实例，用于后续坐标转换
    """
    try:
        # 尝试使用空格作为分隔符加载数据
        try:
            gps_data = np.loadtxt(txt_path, delimiter=' ') # 以空格为分隔符加载
        except ValueError: # 如果失败（可能是因为逗号分隔）
            gps_data = np.loadtxt(txt_path, delimiter=',') # 尝试以逗号为分隔符加载

        if gps_data.ndim == 1: # 如果数据只有一行
            gps_data = gps_data.reshape(1, -1) # 将其转换为二维数组

        assert gps_data.shape[1] >= 4, f"GPS文件需要至少4列 (ts lat lon alt), 找到 {gps_data.shape[1]} 列" # 断言数据列数至少为4

        timestamps = gps_data[:, 0].astype(float) # 提取时间戳并转换为浮点数
        lats = gps_data[:, 1].astype(float) # 提取纬度并转换为浮点数
        lons = gps_data[:, 2].astype(float) # 提取经度并转换为浮点数
        alts = gps_data[:, 3].astype(float) # 提取海拔并转换为浮点数

        # 过滤无效的经纬度值 (纬度范围 [-90, 90], 经度范围 [-180, 180], 且不为0)
        valid_gps_mask = (np.abs(lats) <= 90) & (np.abs(lons) <= 180) & (lats != 0) & (lons != 0)
        if not np.any(valid_gps_mask): # 如果没有任何有效的经纬度坐标
             raise ValueError("GPS数据中没有有效的经纬度坐标。") # 抛出值错误
        initial_count = len(lats) # 记录原始点数
        if np.sum(valid_gps_mask) < initial_count: # 如果有效点数少于原始点数
             print(f"警告：过滤了 {initial_count - np.sum(valid_gps_mask)} 个无效的GPS经纬度点。") # 打印过滤信息

        timestamps = timestamps[valid_gps_mask] # 保留有效时间戳
        lats = lats[valid_gps_mask] # 保留有效纬度
        lons = lons[valid_gps_mask] # 保留有效经度
        alts = alts[valid_gps_mask] # 保留有效海拔
        if len(timestamps) == 0: # 如果过滤后没有数据了
             raise ValueError("过滤无效经纬度后，GPS数据为空。") # 抛出值错误


        # === UTM 投影 ===
        utm_zone_number, utm_hemisphere = auto_utm_projection(lons, lats) # 自动确定UTM分区
        # 构建UTM投影字符串
        proj_string = f"+proj=utm +zone={utm_zone_number}{utm_hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        try:
             projector = Proj(proj_string) # 创建投影对象
        except CRSError as e: # 捕获创建投影对象时的坐标参考系统错误
             raise ValueError(f"无法创建UTM投影 (Zone: {utm_zone_number}{utm_hemisphere}). 请检查经纬度. Proj Error: {e}") # 抛出值错误
        x, y = projector(lons, lats) # 将经纬度投影到UTM坐标 (x, y)
        utm_positions = np.column_stack((x, y, alts)) # 组合成 (N, 3) 的UTM坐标数组 [x, y, alt]

        # === 调用 RANSAC 过滤 GPS 离群点 (在UTM坐标上进行) ===
        print("正在执行 GPS 轨迹 RANSAC 过滤...") # 打印信息
        filtered_times, filtered_utm_positions = filter_gps_outliers_ransac( # 调用RANSAC过滤函数
            timestamps, utm_positions, CONFIG['gps_filtering_ransac'] # 传入时间和UTM位置及配置
        )

        # 记录一下过滤掉的点的数量
        ransac_filtered_count = len(timestamps) - len(filtered_times) # 计算RANSAC过滤掉的点数
        if ransac_filtered_count > 0 : # 如果过滤了点
            print(f"GPS RANSAC 过滤完成，移除了 {ransac_filtered_count} 个点，剩余 {len(filtered_times)} 点。") # 打印信息
        else: # 如果没有过滤点
            print(f"GPS RANSAC 过滤完成，未移除点，剩余 {len(filtered_times)} 点。") # 打印信息


        # 如果过滤后数据点过少，无法进行后续处理 (例如插值至少需要2个点)
        if len(filtered_times) < 2: # 如果过滤后的点数少于2
             raise ValueError("GPS RANSAC 过滤后剩余数据点不足 (< 2)，无法继续处理。请检查 RANSAC 参数设置或原始数据质量。") # 抛出值错误

        # 使用过滤后的数据进行后续处理
        return {
            'timestamps': filtered_times,        # 返回过滤后的时间戳
            'positions': filtered_utm_positions, # 返回过滤后的UTM位置
            'utm_zone': f"{utm_zone_number}{'S' if utm_hemisphere else 'N'}", # 返回UTM分区字符串
            'projector': projector # 返回投影对象
        }
    except FileNotFoundError: # 捕获文件未找到错误
        raise ValueError(f"GPS文件未找到: {txt_path}") # 抛出值错误
    except Exception as e: # 捕获其他GPS数据处理过程中的错误
        print(f"GPS数据加载、投影或过滤失败 ({txt_path}):") # 打印错误信息
        traceback.print_exc() # 打印详细的错误追踪信息
        raise ValueError(f"GPS数据处理失败: {str(e)}") # 抛出值错误

def utm_to_wgs84(utm_points: np.ndarray, projector: Proj) -> np.ndarray:
    """
    将UTM坐标批量转换为WGS84经纬度高程。

    输入:
        utm_points (np.ndarray): UTM坐标数组 (N, 3) [X, Y, Z/Altitude]
        projector (pyproj.Proj): 用于从该UTM带转换回WGS84的pyproj投影仪实例。

    输出:
        np.ndarray: WGS84坐标数组 (N, 3) [longitude, latitude, altitude]
    """
    if utm_points.shape[1] != 3: # 检查UTM点是否为Nx3数组
        raise ValueError("UTM点必须是 Nx3 数组 (X, Y, Z)") # 如果不是则抛出值错误
    if not isinstance(projector, Proj): # 检查projector是否为Proj实例
        raise TypeError("projector 必须是 pyproj.Proj 实例") # 如果不是则抛出类型错误

    # 使用投影对象的inverse=True参数进行反向投影，从UTM转换为经纬度
    lons, lats = projector(utm_points[:, 0], utm_points[:, 1], inverse=True)
    return np.column_stack((lons, lats, utm_points[:, 2])) # 返回组合后的 [经度, 纬度, 高程] 数组

# ----------------------------
# 时间对齐与变换计算
# ----------------------------
def estimate_time_offset(slam_times: np.ndarray, gps_times: np.ndarray, max_samples: int) -> float:
    """
    通过互相关估计SLAM和GPS时间戳之间的时钟偏移。

    输入:
        slam_times (np.ndarray): SLAM轨迹的时间戳数组 (N_slam,)
        gps_times (np.ndarray): GPS轨迹的时间戳数组 (N_gps,)
        max_samples (int): 用于互相关计算的最大样本数。会对两个时间序列进行重采样到此数量。

    输出:
        float: 估计得到的时间偏移量 (秒)。 offset = t_gps - t_slam。
               即，SLAM时间戳加上这个offset后，应与GPS时间戳对齐。
    """
    if len(slam_times) < 2 or len(gps_times) < 2: # 检查SLAM或GPS时间序列是否过短
        print("警告：SLAM或GPS时间序列过短 (<2)，无法进行可靠的时间偏移估计，返回偏移0。") # 打印警告
        return 0.0 # 返回0偏移

    # 确定用于互相关的样本数量，取max_samples、SLAM点数、GPS点数中的最小值
    num_samples = min(max_samples, len(slam_times), len(gps_times))
    # 在SLAM时间范围内均匀采样num_samples个点
    slam_sample_times = np.linspace(slam_times.min(), slam_times.max(), num_samples)
    # 在GPS时间范围内均匀采样num_samples个点
    gps_sample_times = np.linspace(gps_times.min(), gps_times.max(), num_samples)

    # 对采样后的时间序列进行归一化 (减去均值)
    slam_norm = (slam_sample_times - np.mean(slam_sample_times))
    gps_norm = (gps_sample_times - np.mean(gps_sample_times))
    # 计算归一化后时间序列的标准差
    slam_std = np.std(slam_norm)
    gps_std = np.std(gps_norm)

    if slam_std < 1e-9 or gps_std < 1e-9: # 如果标准差过小（时间戳非常集中）
         print("警告：(抽样后)时间戳标准差过小，可能导致互相关不稳定，返回偏移0。") # 打印警告
         return 0.0 # 返回0偏移

    # 用标准差对归一化后的时间序列进行缩放 (使其具有单位方差)
    slam_norm /= slam_std
    gps_norm /= gps_std

    # 计算两个归一化时间序列的互相关，'full'模式会计算所有可能的重叠
    corr = np.correlate(slam_norm, gps_norm, mode='full')
    peak_idx = corr.argmax() # 找到互相关结果中峰值的索引
    # 计算滞后量，表示gps_norm相对于slam_norm的移位数
    lag = peak_idx - len(slam_norm) + 1

    if num_samples <= 1: # 如果用于互相关的样本数不足
         dt_resampled = 0.0 # 重采样时间分辨率设为0
         print("警告: 用于互相关的样本数不足 (<= 1)，无法计算时间分辨率，偏移可能不准确。") # 打印警告
    else: # 如果样本数足够
         # 计算重采样后的时间分辨率 (两个相邻采样点之间的时间差)
         dt_resampled = (slam_sample_times[-1] - slam_sample_times[0]) / (num_samples - 1)

    offset = lag * dt_resampled # 计算最终的时间偏移量 (滞后量 * 时间分辨率)
    print(f"估计的初始时间偏移: {offset:.3f} 秒") # 打印估计的时间偏移
    return offset # 返回时间偏移

def dynamic_time_alignment(slam_data: Dict[str, np.ndarray],
                           gps_data: Dict[str, np.ndarray],
                           time_align_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    动态时间同步系统 (支持中断处理)：
    对每个连续的GPS数据段分别进行三次样条（或线性）插值，将其插值到对应的SLAM时间戳上。

    输入:
        slam_data (Dict[str, np.ndarray]): SLAM数据字典，包含 'timestamps'。
        gps_data (Dict[str, np.ndarray]): GPS数据字典，包含 'timestamps' 和 'positions' (已过滤和投影的UTM坐标)。
        time_align_config (Dict[str, Any]): 时间对齐相关的配置参数，如 'max_samples_for_corr', 'max_gps_gap_threshold'。

    输出:
        Tuple[np.ndarray, np.ndarray]:
            - aligned_gps_full (np.ndarray): (N_slam, 3) 数组。在每个SLAM时间戳上插值得到的GPS位置。
                                             对于无法插值的SLAM时间戳，对应位置为 [NaN, NaN, NaN]。
            - valid_mask (np.ndarray): (N_slam,) 布尔数组。标记哪些SLAM时间戳成功获得了有效的插值GPS位置。
    """
    slam_times = slam_data['timestamps'] # 获取SLAM时间戳
    gps_times = gps_data['timestamps']     # 获取过滤后的GPS时间戳
    gps_positions = gps_data['positions'] # 获取过滤后的UTM位置
    max_corr_samples = time_align_config['max_samples_for_corr'] # 获取用于时间偏移估计的最大样本数
    max_gps_gap_threshold = time_align_config['max_gps_gap_threshold'] # 获取GPS时间中断的阈值

    n_slam = len(slam_times) # SLAM时间戳数量
    n_gps = len(gps_times) # GPS时间戳数量

    # 初始化对齐后的GPS位置数组，填充为NaN，大小与SLAM轨迹相同
    aligned_gps_full = np.full((n_slam, 3), np.nan)
    # 初始化有效掩码数组，全为False
    valid_mask = np.zeros(n_slam, dtype=bool)

    if n_slam == 0 or n_gps < 2: # 如果SLAM时间戳为空，或者GPS点数少于2个（无法插值）
        print(f"警告：SLAM 时间戳为空 ({n_slam}) 或 有效GPS点不足 ({n_gps} < 2)，无法进行时间对齐。") # 打印警告
        return aligned_gps_full, valid_mask # 返回空的对齐结果和掩码

    # 估计初始的时间偏移
    offset = estimate_time_offset(slam_times, gps_times, max_corr_samples)
    adjusted_gps_times = gps_times + offset # 调整GPS时间戳，使其与SLAM时间戳初步对齐

    try:
        # 对调整后的GPS时间戳及其对应的位置进行排序
        sorted_indices = np.argsort(adjusted_gps_times)
        adjusted_gps_times_sorted = adjusted_gps_times[sorted_indices]
        gps_positions_sorted = gps_positions[sorted_indices]

        # 去除调整后GPS时间戳中的重复项，保留第一个出现的
        unique_times, unique_indices = np.unique(adjusted_gps_times_sorted, return_index=True)
        n_unique_gps = len(unique_times) # 去重后的GPS时间戳数量

        if n_unique_gps < 2: # 如果去重后GPS点数少于2
             print("警告：去重后的有效GPS时间戳少于2个点，无法进行插值。") # 打印警告
             return aligned_gps_full, valid_mask # 返回

        if n_unique_gps < n_gps: # 如果去重操作移除了点
            print(f"警告：移除了 {n_gps - n_unique_gps} 个重复的GPS时间戳。") # 打印信息
            adjusted_gps_times_sorted = unique_times # 更新为去重后的时间戳
            gps_positions_sorted = gps_positions_sorted[unique_indices] # 更新为去重后的位置

        # 计算排序后GPS时间戳之间的时间差
        time_diffs = np.diff(adjusted_gps_times_sorted)
        # 找到时间差大于中断阈值的间隙的结束索引 (这些索引指向间隙前的那个点)
        gap_indices = np.where(time_diffs > max_gps_gap_threshold)[0]

        # 根据找到的间隙，将GPS数据划分为多个连续的段
        # segment_starts 存储每个段的起始索引 (在 adjusted_gps_times_sorted 中的索引)
        segment_starts = [0] + (gap_indices + 1).tolist() # 第一个段从0开始，后续段从间隙后一个点开始
        # segment_ends 存储每个段的结束索引
        segment_ends = gap_indices.tolist() + [n_unique_gps - 1] # 间隙前的点是前一段的结束，最后一个点是最后一段的结束

        print(f"检测到 {len(gap_indices)} 个 GPS 时间中断 (阈值 > {max_gps_gap_threshold:.1f}s)，将数据分为 {len(segment_starts)} 段进行插值。") # 打印分段信息

        total_valid_points = 0 # 初始化成功插值的点数计数器
        for i in range(len(segment_starts)): # 遍历每个GPS数据段
            start_idx = segment_starts[i] # 当前段的起始索引
            end_idx = segment_ends[i] # 当前段的结束索引
            segment_len = end_idx - start_idx + 1 # 当前段的长度（点数）

            if segment_len < 4: # 如果段内点数不足4个
                kind = 'linear' if segment_len >= 2 else None # 点数>=2用线性插值，否则无法插值
                if kind is None: continue # 如果无法插值，跳过此段
            else: # 如果点数>=4
                kind = 'cubic' # 使用三次样条插值

            segment_times = adjusted_gps_times_sorted[start_idx : end_idx + 1] # 获取当前段的时间戳
            segment_positions = gps_positions_sorted[start_idx : end_idx + 1] # 获取当前段的位置数据
            segment_min_time = segment_times[0] # 当前段的最小时间
            segment_max_time = segment_times[-1] # 当前段的最大时间

            try:
                if not np.all(np.diff(segment_times) > 0): # 检查段内时间戳是否严格单调递增
                     print(f"警告：段 {i+1} 内时间戳非严格单调递增，跳过此段。") # 如果不是，打印警告并跳过
                     continue

                # 创建一维插值函数，针对当前段的时间和位置数据
                interp_func_segment = interp1d(
                    segment_times, segment_positions, axis=0, kind=kind, # axis=0表示对每一列（x,y,z）分别插值
                    bounds_error=False, fill_value=np.nan # 超出范围不报错，填充NaN
                )
            except ValueError as e: # 捕获创建插值函数时的错误
                print(f"警告：为段 {i+1} 创建插值函数失败 ({kind}插值, {segment_len}点): {e}。跳过此段。") # 打印警告并跳过
                continue

            # 找到落在当前GPS段时间范围内的SLAM时间戳的索引
            slam_indices_in_segment = np.where(
                (slam_times >= segment_min_time) & (slam_times <= segment_max_time)
            )[0]

            if len(slam_indices_in_segment) > 0: # 如果有SLAM时间戳落在此段
                # 对这些SLAM时间戳进行插值，得到对应的GPS位置
                interpolated_positions = interp_func_segment(slam_times[slam_indices_in_segment])
                # 将插值结果存入 aligned_gps_full 数组的对应位置
                aligned_gps_full[slam_indices_in_segment] = interpolated_positions
                # 标记那些成功插值（非NaN）的SLAM时间戳为有效
                non_nan_mask_segment = ~np.isnan(interpolated_positions).any(axis=1) # 检查每一行是否包含NaN
                valid_mask[slam_indices_in_segment[non_nan_mask_segment]] = True # 更新全局有效掩码
                total_valid_points += np.sum(non_nan_mask_segment) # 累加有效插值点数

        print(f"分段插值完成：在 {n_slam} 个 SLAM 时间点上共生成了 {total_valid_points} 个有效的对齐GPS位置。") # 打印插值结果总结
        if total_valid_points == 0: # 如果没有生成任何有效的对齐点
             print("警告：没有生成任何有效的对齐GPS位置，请检查时间范围重叠、中断阈值或数据质量。") # 打印警告

        return aligned_gps_full, valid_mask # 返回对齐后的GPS位置和有效掩码

    except ValueError as e: # 捕获时间对齐或分段插值过程中的值错误
        print(f"时间对齐或分段插值过程中发生错误: {e}.") # 打印错误信息
        traceback.print_exc() # 打印详细的错误追踪信息
        return np.full((n_slam, 3), np.nan), np.zeros(n_slam, dtype=bool) # 返回空的对齐结果和掩码

def compute_sim3_transform_robust(src: np.ndarray, dst: np.ndarray,
                                 min_samples: int, residual_threshold: float,
                                 max_trials: int, min_inliers_needed: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    使用 RANSAC 稳健地估计源点集(src)到目标点集(dst)的 Sim3 变换 (旋转R, 平移t, 尺度s)。
    该函数首先使用RANSAC筛选内点，然后基于这些内点计算Sim3变换。

    输入:
        src (np.ndarray): 源点云坐标数组 (N, 3)
        dst (np.ndarray): 目标点云坐标数组 (N, 3)，与src中的点一一对应
        min_samples (int): RANSAC拟合模型所需的最小样本数
        residual_threshold (float): RANSAC内点判断的残差阈值
        max_trials (int): RANSAC最大迭代次数
        min_inliers_needed (int): 成功计算Sim3变换所需的最小内点数量

    输出:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
            - R (Optional[np.ndarray]): 3x3 旋转矩阵。如果失败则为 None。
            - t (Optional[np.ndarray]): 3x1 平移向量。如果失败则为 None。
            - scale (Optional[float]): 尺度因子。如果失败则为 None。
    """
    if src.shape[0] < min_samples or dst.shape[0] < min_samples: # 检查点数是否满足RANSAC最小样本要求
        print(f"错误: 点数不足 ({src.shape[0]}) ，无法进行 Sim3 RANSAC (需要至少 {min_samples} 个)") # 打印错误
        return None, None, None # 返回None表示失败
    if src.shape != dst.shape: # 检查源点和目标点数量或维度是否一致
         print("错误: Sim3 RANSAC：源点和目标点数量或维度不一致") # 打印错误
         return None, None, None # 返回None

    # --- 对点集进行归一化，以提高RANSAC的数值稳定性 ---
    src_mean = np.mean(src, axis=0) # 计算源点云的质心
    dst_mean = np.mean(dst, axis=0) # 计算目标点云的质心
    src_centered = src - src_mean # 源点云中心化
    dst_centered = dst - dst_mean # 目标点云中心化
    # 计算源点云到其质心的平均距离，作为归一化因子
    src_norm_factor = np.mean(np.linalg.norm(src_centered, axis=1))
    # 计算目标点云到其质心的平均距离，作为归一化因子
    dst_norm_factor = np.mean(np.linalg.norm(dst_centered, axis=1))

    if src_norm_factor < 1e-9 or dst_norm_factor < 1e-9: # 如果归一化因子过小（点集非常集中）
        print("警告：Sim3 RANSAC 输入点集分布非常集中，归一化可能不稳定。跳过归一化。") # 打印警告
        src_normalized = src_centered # 使用中心化后的点
        dst_normalized = dst_centered # 使用中心化后的点
    else: # 否则进行归一化
        src_normalized = src_centered / src_norm_factor # 归一化源点云
        dst_normalized = dst_centered / dst_norm_factor # 归一化目标点云

    try:
        # 为目标点云的X, Y, Z三个维度分别创建RANSAC回归器
        # 这里的逻辑是：假设存在一个仿射变换 T (近似Sim3中的旋转和尺度部分)
        #使得 dst_normalized_i = T @ src_normalized_i + noise
        # RANSAC会尝试找到这个 T
        ransac_x = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, stop_probability=0.99)
        ransac_y = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, stop_probability=0.99)
        ransac_z = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, stop_probability=0.99)

        # 分别对X, Y, Z维度进行拟合
        ransac_x.fit(src_normalized, dst_normalized[:, 0]) # 拟合X维度
        ransac_y.fit(src_normalized, dst_normalized[:, 1]) # 拟合Y维度
        ransac_z.fit(src_normalized, dst_normalized[:, 2]) # 拟合Z维度

        # 获取三个维度共同的内点掩码 (要求一个点在X,Y,Z三个维度上都是内点)
        inlier_mask = ransac_x.inlier_mask_ & ransac_y.inlier_mask_ & ransac_z.inlier_mask_
        num_inliers = np.sum(inlier_mask) # 计算内点数量
        print(f"Sim3 RANSAC: 找到 {num_inliers} / {src.shape[0]} 个内点 (阈值={residual_threshold}, 迭代={max_trials})") # 打印RANSAC结果

        if num_inliers < min_inliers_needed: # 如果内点数量不足以计算可靠的Sim3变换
            print(f"错误: Sim3 RANSAC: 有效内点不足 ({num_inliers} < {min_inliers_needed})，无法计算可靠的 Sim3 变换") # 打印错误
            return None, None, None # 返回None

        # 使用RANSAC筛选出的内点来计算最终的Sim3变换
        return compute_sim3_transform(src[inlier_mask], dst[inlier_mask])

    except ValueError as ve: # 捕获RANSAC过程中的值错误
         print(f"Sim3 RANSAC 失败: {ve}.") # 打印错误
         return None, None, None # 返回None
    except Exception as e: # 捕获其他未知错误
         print(f"Sim3 RANSAC 过程中发生未知错误: {e}.") # 打印错误
         traceback.print_exc() # 打印详细错误追踪
         return None, None, None # 返回None

def compute_sim3_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    计算从源点集(src)到目标点集(dst)的最佳 Sim3 变换 (旋转R, 平移t, 尺度s)。
    这通常在已经通过RANSAC等方法筛选出内点后调用。

    输入:
        src (np.ndarray): 源点云坐标数组 (N, 3), N >= 3
        dst (np.ndarray): 目标点云坐标数组 (N, 3), 与src中的点一一对应

    输出:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
            - R (Optional[np.ndarray]): 3x3 旋转矩阵。如果失败则为 None。
            - t (Optional[np.ndarray]): 3x1 平移向量。如果失败则为 None。
            - scale (Optional[float]): 尺度因子。如果失败则为 None。
    """
    if src.shape[0] < 3: # 计算Sim3变换至少需要3个点对
         print(f"错误: 计算 Sim3 变换需要至少 3 个点，但只有 {src.shape[0]} 个内点") # 打印错误
         return None, None, None # 返回None
    if src.shape != dst.shape: # 检查源点和目标点数量或维度是否一致
         print("错误: Sim3 计算：源点和目标点数量或维度不一致") # 打印错误
         return None, None, None # 返回None

    try:
        # 1. 计算质心
        src_centroid = np.mean(src, axis=0) # 源点云质心
        dst_centroid = np.mean(dst, axis=0) # 目标点云质心

        # 2. 点云中心化
        src_centered = src - src_centroid # 源点云去中心化
        dst_centered = dst - dst_centroid # 目标点云去中心化

        # 3. 计算协方差矩阵 H = src_centered^T * dst_centered
        H = src_centered.T @ dst_centered

        # 4. 对 H 进行奇异值分解 (SVD): H = U * S * V^T
        U, S, Vt = np.linalg.svd(H)
        V = Vt.T # V = (V^T)^T

        # 5. 计算旋转矩阵 R = V * U^T
        R = V @ U.T

        # 确保 R 是一个正常的旋转矩阵 (行列式为 +1)
        if np.linalg.det(R) < 0: # 如果行列式为负，说明得到了一个反射矩阵
            print("警告: 检测到反射（行列式为负），修正旋转矩阵。") # 打印警告
            Vt_copy = Vt.copy() # 复制Vt以进行修改
            Vt_copy[-1, :] *= -1 # 将V的最后一列（或U的最后一列，取决于实现）取反
            R = Vt_copy.T @ U.T # 重新计算R

        # 6. 计算尺度因子 s
        # s = sum( dot(dst_centered_i, src_centered_i @ R^T) ) / sum( ||src_centered_i||^2 )
        # 更简洁的写法: s = trace(R^T @ H) / trace(src_centered^T @ src_centered)
        # 或者使用Arun方法中的分子/分母形式
        src_dist_sq = np.sum(src_centered**2, axis=1) # 计算每个中心化源点到其质心的距离平方
        # 分子: sum over i ( dst_centered_i^T * R * src_centered_i )
        # 这里用的是 sum ( dst_centered_i . (src_centered_i @ R^T) )
        #  = sum ( sum_j (dst_centered_ij * (src_centered_i @ R^T)_j ) )
        numerator = np.sum(np.sum(dst_centered * (src_centered @ R.T), axis=1))
        denominator = np.sum(src_dist_sq) # 分母: 中心化源点云的方差和

        if denominator < 1e-9: # 如果分母过小（源点非常集中）
             print("警告：源点非常集中于质心，无法可靠计算尺度，默认为 1.0") # 打印警告
             scale = 1.0 # 尺度设为1
        else:
            scale = numerator / denominator # 计算尺度
            if scale <= 1e-6: # 如果计算出的尺度非常小
                 print(f"警告：计算出的尺度非常小 ({scale:.2e})，可能存在问题。重置为 1.0") # 打印警告
                 scale = 1.0 # 重置尺度为1

        # 7. 计算平移向量 t = dst_centroid - s * R @ src_centroid
        t = dst_centroid - scale * (R @ src_centroid)
        print(f"计算得到的 Sim3 参数: scale={scale:.4f}") # 打印计算得到的尺度
        return R, t, scale # 返回旋转矩阵、平移向量和尺度因子

    except np.linalg.LinAlgError as e: # 捕获线性代数计算错误
        print(f"错误: 计算 Sim3 变换时发生线性代数错误: {e}") # 打印错误
        return None, None, None # 返回None
    except Exception as e: # 捕获其他未知错误
        print(f"错误: 计算 Sim3 变换时发生未知错误: {e}") # 打印错误
        traceback.print_exc() # 打印详细错误追踪
        return None, None, None # 返回None

def transform_trajectory(positions: np.ndarray, quaternions: np.ndarray,
                        R: np.ndarray, t: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用Sim3变换 (旋转R, 平移t, 尺度s) 到整个轨迹（位置和姿态）。

    输入:
        positions (np.ndarray): 原始轨迹的位置数组 (N, 3) [x, y, z]
        quaternions (np.ndarray): 原始轨迹的姿态四元数数组 (N, 4) [x, y, z, w]
        R (np.ndarray): 3x3 旋转矩阵 (从Sim3变换得到)
        t (np.ndarray): 3x1 平移向量 (从Sim3变换得到)
        scale (float): 尺度因子 (从Sim3变换得到)

    输出:
        Tuple[np.ndarray, np.ndarray]:
            - trans_pos (np.ndarray): 变换后的位置数组 (N, 3)
            - trans_quat (np.ndarray): 变换后的姿态四元数数组 (N, 4)
    """
    # 变换位置: P_transformed = s * R * P_original + t
    # 注意：这里 R 是应用到原始坐标系，所以是 P_original @ R.T (如果R是从原始到目标)
    # 或者 R @ P_original.T 然后再转置回来。
    # 如果 R 是将 src 坐标系下的向量旋转到 dst 坐标系，则 P_dst = s * R @ P_src + t
    # 如果 positions 是 (N,3)， R 是 (3,3)，则 s * (positions @ R.T) + t
    trans_pos = scale * (positions @ R.T) + t # 应用Sim3变换到位置

    # 变换姿态 (四元数): Q_transformed = Q_sim3_rotation * Q_original
    R_sim3_rot = Rotation.from_matrix(R) # 从Sim3的旋转矩阵创建Rotation对象
    trans_quat_list = [] # 初始化变换后的四元数列表
    for q_xyzw in quaternions: # 遍历原始轨迹中的每个四元数
        original_rot = Rotation.from_quat(q_xyzw) # 从原始四元数创建Rotation对象
        new_rot = R_sim3_rot * original_rot # 将Sim3的旋转左乘到原始旋转上
        new_quat_xyzw = new_rot.as_quat() # 将新的旋转对象转换为四元数
        trans_quat_list.append(new_quat_xyzw) # 添加到列表

    return trans_pos, np.array(trans_quat_list) # 返回变换后的位置和四元数数组

def plot_results(original_pos: np.ndarray, corrected_pos: np.ndarray,
                 gps_pos: np.ndarray,
                 valid_indices_for_error: np.ndarray,
                 aligned_gps_for_error: np.ndarray,
                 slam_times: np.ndarray,
                 ekf_errors: Optional[np.ndarray] = None
                ) -> None:
    """
    增强的可视化系统，显示原始轨迹、修正后轨迹、GPS点和误差。

    输入:
        original_pos (np.ndarray): 原始SLAM轨迹的位置数据 (N_slam, 3)
        corrected_pos (np.ndarray): EKF修正后的SLAM轨迹的位置数据 (N_slam, 3)
        gps_pos (np.ndarray): 原始(过滤后)的GPS UTM点 (N_gps_filtered, 3)
        valid_indices_for_error (np.ndarray): 用于误差评估的有效SLAM时间戳的索引 (M,)
        aligned_gps_for_error (np.ndarray): 与上述有效SLAM时间戳对齐的GPS位置 (M, 3)
        slam_times (np.ndarray): SLAM轨迹的时间戳 (N_slam,)
        ekf_errors (Optional[np.ndarray]): EKF最终误差数组 (M,), EKF修正后轨迹与对齐GPS点之间的距离。
                                          如果为None或为空，则不绘制误差相关图。
    输出:
        None: 直接显示绘图结果。
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置matplotlib默认字体为SimHei以支持中文
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
    except: # 如果设置字体失败
        print("警告：未能设置中文字体 'SimHei'。标签可能显示为方块。请确保已安装该字体。") # 打印警告

    plt.figure(figsize=(15, 12)) # 创建一个新的图形，设置大小
    plt.suptitle("SLAM-GPS 轨迹对齐与融合结果", fontsize=16) # 设置图形总标题

    # --- 1. 2D轨迹对比 (XY平面) ---
    ax1 = plt.subplot(2, 2, 1) # 创建2x2子图中的第一个
    ax1.plot(original_pos[:, 0], original_pos[:, 1], 'b--', alpha=0.6, linewidth=1, label='原始 SLAM 轨迹') # 绘制原始SLAM轨迹 (蓝色虚线)
    ax1.plot(corrected_pos[:, 0], corrected_pos[:, 1], 'g-', linewidth=1.5, label='EKF 修正后轨迹') # 绘制EKF修正后轨迹 (绿色实线)
    ax1.scatter(gps_pos[:, 0], gps_pos[:, 1], c='r', marker='x', s=30, label='GPS 参考点 (过滤后, UTM)') # 绘制过滤后的GPS点 (红色x)
    step = max(1, len(valid_indices_for_error) // 100) # 控制绘制对齐GPS点的密度，最多100个点
    if len(valid_indices_for_error) > 0: # 如果有用于误差评估的对齐GPS点
       # 绘制对齐的GPS点 (橙色空心圆圈)，用于显示误差评估的参考
       ax1.scatter(aligned_gps_for_error[::step, 0], aligned_gps_for_error[::step, 1],
                   facecolors='none', edgecolors='orange', marker='o', s=40,
                   label='对齐GPS点 (插值, 用于误差评估)')

    ax1.set_title('轨迹对比 (XY平面)') # 设置子图标题
    ax1.set_xlabel('X (米)') # 设置X轴标签
    ax1.set_ylabel('Y (米)') # 设置Y轴标签
    ax1.legend() # 显示图例
    ax1.grid(True) # 显示网格
    ax1.axis('equal') # 设置XY轴等比例，使轨迹形状不失真

    # --- 2. 3D轨迹对比 ---
    ax3d = plt.subplot(2, 2, 2, projection='3d') # 创建2x2子图中的第二个，指定为3D投影
    ax3d.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 'b--', alpha=0.6, linewidth=1, label='原始 SLAM 轨迹') # 绘制3D原始SLAM轨迹
    ax3d.plot(corrected_pos[:, 0], corrected_pos[:, 1], corrected_pos[:, 2], 'g-', linewidth=1.5, label='EKF 修正后轨迹') # 绘制3D EKF修正后轨迹
    ax3d.scatter(gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2], c='r', marker='x', s=30, label='GPS 参考点 (过滤后, UTM)') # 绘制3D GPS点
    if len(valid_indices_for_error) > 0: # 如果有对齐的GPS点
        ax3d.scatter(aligned_gps_for_error[::step, 0], aligned_gps_for_error[::step, 1], aligned_gps_for_error[::step, 2],
                    facecolors='none', edgecolors='orange', marker='o', s=40,
                    label='对齐GPS点 (插值)') # 绘制3D对齐GPS点

    ax3d.set_title('轨迹对比 (3D)') # 设置子图标题
    ax3d.set_xlabel('X (米)') # X轴标签
    ax3d.set_ylabel('Y (米)') # Y轴标签
    ax3d.set_zlabel('Z (米)') # Z轴标签
    ax3d.legend() # 显示图例
    try: # 尝试自动调整3D视图的范围，使其更美观
        # 计算修正后轨迹在X,Y,Z三个方向上的最大范围的一半，并稍作放大
        max_range = np.array([corrected_pos[:,0].max()-corrected_pos[:,0].min(),
                              corrected_pos[:,1].max()-corrected_pos[:,1].min(),
                              corrected_pos[:,2].max()-corrected_pos[:,2].min()]).max() / 2.0 * 1.1
        if max_range < 1.0: max_range = 5.0 # 保证最小范围
        mid_x = np.median(corrected_pos[:,0]) # 计算X方向中位数作为中心
        mid_y = np.median(corrected_pos[:,1]) # 计算Y方向中位数作为中心
        mid_z = np.median(corrected_pos[:,2]) # 计算Z方向中位数作为中心
        ax3d.set_xlim(mid_x - max_range, mid_x + max_range) # 设置X轴范围
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range) # 设置Y轴范围
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range) # 设置Z轴范围
    except: # 如果自动调整失败，则忽略
        pass

    # --- 3. 最终误差分布 (直方图) ---
    ax_hist = plt.subplot(2, 2, 3) # 创建2x2子图中的第三个
    if ekf_errors is not None and len(ekf_errors) > 0: # 如果有有效的EKF误差数据
        mean_err = np.mean(ekf_errors) # 计算平均误差
        std_err = np.std(ekf_errors) # 计算误差标准差
        max_err = np.max(ekf_errors) # 计算最大误差
        median_err = np.median(ekf_errors) # 计算误差中位数

        ax_hist.hist(ekf_errors, bins=30, alpha=0.75, color='purple') # 绘制误差直方图
        ax_hist.axvline(mean_err, color='red', linestyle='dashed', linewidth=1, label=f'均值: {mean_err:.2f}m') # 绘制平均误差线
        ax_hist.axvline(median_err, color='orange', linestyle='dashed', linewidth=1, label=f'中位数: {median_err:.2f}m') # 绘制中位数误差线

        ax_hist.set_title(f'最终位置误差分布 (N={len(ekf_errors)})\n标准差: {std_err:.2f}m, 最大值: {max_err:.2f}m') # 设置子图标题
        ax_hist.set_xlabel('误差 (米)') # X轴标签
        ax_hist.set_ylabel('频数') # Y轴标签
        ax_hist.legend() # 显示图例
        ax_hist.grid(True) # 显示网格
    else: # 如果没有误差数据
        ax_hist.set_title("最终位置误差分布") # 设置标题
        ax_hist.text(0.5, 0.5, "无有效误差数据", ha='center', va='center', fontsize=12) # 显示提示信息
        ax_hist.set_xlabel('误差 (米)') # X轴标签
        ax_hist.set_ylabel('频数') # Y轴标签

    # --- 4. 误差随时间变化图 ---
    ax_err_time = plt.subplot(2, 2, 4) # 创建2x2子图中的第四个
    # 检查是否有EKF误差数据、用于误差评估的有效索引、以及SLAM时间戳与修正后位置点数是否一致
    if ekf_errors is not None and len(valid_indices_for_error) > 0 and len(slam_times) == len(corrected_pos):
        valid_timestamps = slam_times[valid_indices_for_error] # 获取发生误差评估的那些SLAM时间戳
        if len(valid_timestamps) == len(ekf_errors): # 确保这些时间戳的数量与误差数量一致
             relative_time = valid_timestamps - valid_timestamps[0] # 计算相对时间（从第一个有效时间点开始）
             ax_err_time.plot(relative_time, ekf_errors, 'r-', linewidth=1, alpha=0.8, label='绝对位置误差') # 绘制误差随时间变化曲线
             ax_err_time.set_xlabel('相对时间 (秒)') # X轴标签
             ax_err_time.set_ylabel('误差 (米)') # Y轴标签
             ax_err_time.set_title('误差随时间变化') # 子图标题
             ax_err_time.grid(True) # 显示网格
             ax_err_time.legend() # 显示图例
             ax_err_time.set_ylim(bottom=0) # 设置Y轴从0开始，因为误差通常不为负
        else: # 如果时间戳与误差数量不匹配
             ax_err_time.set_title("误差随时间变化") # 设置标题
             ax_err_time.text(0.5, 0.5, "时间戳与误差数量不匹配", ha='center', va='center', fontsize=10) # 显示提示
             ax_err_time.set_xlabel('相对时间 (秒)') # X轴标签
             ax_err_time.set_ylabel('误差 (米)') # Y轴标签

    else: # 如果缺少绘制误差随时间图所需的数据
         ax_err_time.set_title("误差随时间变化") # 设置标题
         ax_err_time.text(0.5, 0.5, "无有效误差数据或时间戳", ha='center', va='center', fontsize=10) # 显示提示
         ax_err_time.set_xlabel('相对时间 (秒)') # X轴标签
         ax_err_time.set_ylabel('误差 (米)') # Y轴标签


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 自动调整子图布局，避免重叠，并为总标题留出空间
    plt.show() # 显示图形


def select_file_dialog(title: str, filetypes: List[Tuple[str, str]]) -> str:
    """
    通用的文件选择对话框。

    输入:
        title (str): 文件选择对话框的标题。
        filetypes (List[Tuple[str, str]]): 文件类型过滤器列表，例如 [("文本文件", "*.txt"), ("所有文件", "*.*")]。

    输出:
        str: 用户选择的文件路径。如果用户取消选择，则返回空字符串。
    """
    root = tk.Tk() # 创建一个Tkinter根窗口 (主窗口)
    root.withdraw() # 隐藏主窗口，我们只需要对话框
    # 打开文件选择对话框，并获取用户选择的文件路径
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy() # 销毁Tkinter根窗口
    return file_path if file_path else "" # 如果用户选择了文件则返回路径，否则返回空字符串

def select_slam_file() -> str:
    """
    打开一个文件选择对话框，让用户选择SLAM轨迹文件。

    输入:
        无

    输出:
        str: 用户选择的SLAM轨迹文件路径。如果取消，则为空字符串。
    """
    return select_file_dialog( # 调用通用的文件选择对话框函数
        "选择 SLAM 轨迹文件 (TUM格式: timestamp tx ty tz qx qy qz qw)", # 对话框标题
        [("文本文件", "*.txt"), ("所有文件", "*.*")] # 文件类型过滤器
    )

def select_gps_file() -> str:
    """
    打开一个文件选择对话框，让用户选择GPS数据文件。

    输入:
        无

    输出:
        str: 用户选择的GPS数据文件路径。如果取消，则为空字符串。
    """
    return select_file_dialog( # 调用通用的文件选择对话框函数
        "选择 GPS 数据文件 (格式: timestamp lat lon alt [...], 空格或逗号分隔)", # 对话框标题
        [("文本文件", "*.txt"), ("CSV文件", "*.csv"),("所有文件", "*.*")] # 文件类型过滤器
    )

# ----------------------------
# EKF 实现
# ----------------------------

class ExtendedKalmanFilter:
    def __init__(self, initial_pos: np.ndarray, initial_quat: np.ndarray,
                 config: Dict[str, Any]):
        """
        初始化扩展卡尔曼滤波器 (实现航位推算+平滑恢复逻辑)。

        状态向量 (self.state): [x, y, z, qx, qy, qz, qw] (7维)
                            位置 (x,y,z) 和 姿态四元数 (qx,qy,qz,qw, scipy格式)

        输入:
            initial_pos (np.ndarray): 初始位置 [x, y, z]
            initial_quat (np.ndarray): 初始姿态四元数 [x, y, z, w]
            config (Dict[str, Any]): EKF相关的配置参数字典，通常是 CONFIG['ekf']
        """
        if not (initial_pos.shape == (3,) and initial_quat.shape == (4,)): # 检查初始位置和四元数的维度
            raise ValueError(f"EKF 初始化: 初始位置或四元数维度错误") # 如果维度错误，抛出值错误

        ekf_config = config # 直接使用传入的EKF配置

        initial_quat_normalized = self.normalize_quaternion(initial_quat) # 归一化初始四元数
        # 初始化状态向量，拼接位置和归一化后的四元数
        self.state = np.concatenate([initial_pos, initial_quat_normalized]).astype(float)
        # 初始化协方差矩阵，使用配置中的初始协方差对角线元素
        self.cov = np.diag(ekf_config['initial_cov_diag']).astype(float)
        # 初始化过程噪声协方差矩阵Q (每秒)，使用配置中的过程噪声对角线元素
        self.Q_per_sec = np.diag(ekf_config['process_noise_diag']).astype(float)
        # 初始化测量噪声协方差矩阵R，使用配置中的测量噪声对角线元素
        self.R = np.diag(ekf_config['meas_noise_diag']).astype(float)

        # 维度检查，确保内部变量符合预期
        if self.state.shape != (7,): raise ValueError(f"EKF: 内部状态维度错误")
        if self.cov.shape != (7, 7): raise ValueError(f"EKF: 内部协方差维度错误")
        if self.Q_per_sec.shape != (7, 7): raise ValueError(f"EKF: 内部过程噪声Q维度错误")
        if self.R.shape != (3, 3): raise ValueError(f"EKF: 内部测量噪声R维度错误")

        # --- 新增: 平滑状态相关变量 ---
        self.gnss_available_prev = None # 上一时刻的GNSS可用性状态
        self.gnss_update_weight = 0.0 # GNSS 更新贡献的权重 (0.0 到 1.0)，用于平滑过渡
        # 从配置中读取平滑过渡所需的步数，确保至少为1步
        self.transition_steps = max(1, int(ekf_config.get('transition_steps', 10)))
        # 计算每一步权重增加的量
        self.weight_delta = 1.0 / self.transition_steps if self.transition_steps > 0 else 1.0
        self._last_predicted_state = self.state.copy() # 存储上一次的预测状态，用于平滑插值

    @staticmethod
    def normalize_quaternion(q: np.ndarray) -> np.ndarray:
        """
        归一化四元数。

        输入:
            q (np.ndarray): 待归一化的四元数 [x, y, z, w]

        输出:
            np.ndarray: 归一化后的四元数 [x, y, z, w]。如果输入为零四元数，则返回单位四元数 [0,0,0,1]。
        """
        norm = np.linalg.norm(q) # 计算四元数的模长
        if norm < 1e-9: # 如果模长过小（接近零）
            print("警告：尝试归一化零四元数，返回默认单位四元数 [0,0,0,1]。") # 打印警告
            return np.array([0.0, 0.0, 0.0, 1.0]) # 返回单位四元数
        return q / norm # 返回归一化后的四元数

    def _predict(self, slam_motion_update: Tuple[np.ndarray, np.ndarray], delta_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF 预测步骤：使用 SLAM 相对运动更新状态和协方差。

        输入:
            slam_motion_update (Tuple[np.ndarray, np.ndarray]):
                - delta_pos_local (np.ndarray): SLAM在上一帧局部坐标系下的位置变化 [dx, dy, dz]
                - delta_quat (np.ndarray): SLAM的相对旋转四元数 [dqx, dqy, dqz, dqw]
            delta_time (float): 当前预测步与上一步之间的时间间隔 (秒)

        输出:
            Tuple[np.ndarray, np.ndarray]:
                - predicted_state (np.ndarray): 预测后的状态向量 [x, y, z, qx, qy, qz, qw]
                - predicted_covariance (np.ndarray): 预测后的协方差矩阵 (7x7)
        """
        # --- 1. 获取上一时刻状态 ---
        prev_state = self.state # 上一时刻的状态向量
        prev_cov = self.cov # 上一时刻的协方差矩阵
        prev_pos = prev_state[:3] # 上一时刻的位置
        prev_quat = prev_state[3:] # 上一时刻的姿态四元数
        prev_rot = Rotation.from_quat(prev_quat) # 上一时刻的旋转对象

        delta_pos_local, delta_quat = slam_motion_update # 解包SLAM的相对运动
        delta_rot = Rotation.from_quat(delta_quat) # SLAM的相对旋转对象

        # --- 2. 状态预测 (运动模型) ---
        # 位置预测: pos_pred = pos_prev + R(q_prev) @ delta_pos_local
        # 将局部坐标系下的位置变化，通过上一时刻的姿态旋转到世界坐标系，再加到上一时刻的世界坐标位置上
        predicted_pos = prev_pos + prev_rot.apply(delta_pos_local)
        # 姿态预测: q_pred = q_prev * delta_q (四元数乘法表示旋转的叠加)
        predicted_rot = prev_rot * delta_rot # 预测的旋转对象
        predicted_quat = predicted_rot.as_quat() # 转换为四元数

        predicted_state = np.concatenate([predicted_pos, predicted_quat]) # 拼接成预测的状态向量

        # --- 3. 计算状态转移矩阵 Jacobian F (7x7) ---
        # F = ∂f/∂x | evaluated at prev_state, motion_update
        # 对于7D状态（位置+四元数），完整的雅可比矩阵比较复杂。
        # 这里使用一个简化的近似：
        #   - 位置部分对位置的偏导是单位阵 I (3x3)。
        #   - 位置部分对姿态的偏导，以及姿态部分对姿态的偏导，都涉及到旋转矩阵或四元数乘法对四元数的复杂导数。
        #   - 姿态部分对位置的偏导是零矩阵 0 (4x3)。
        # 简化处理：假设 F 近似为单位矩阵，或者只考虑主要影响项。
        # 更精确的模型会计算 d(R@delta_p)/dq 和 d(q*delta_q)/dq。
        # 此处，为了简化，我们先用单位矩阵作为F，并将未建模的耦合效应吸收到过程噪声Q中。
        F = np.eye(7) # 状态转移矩阵雅可比，此处简化为单位阵

        # (可选的更精确F的考虑，但会增加复杂性)
        # 例如，可以加入旋转对位置预测的影响项 ∂(R(q_prev)@delta_pos_local)/∂q_prev
        # 这涉及到旋转矩阵对四元数的导数，通常用反对称矩阵表示。
        # rotated_delta_pos = prev_rot.apply(delta_pos_local)
        # x, y, z = rotated_delta_pos
        # skew_rotated_delta_pos = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        # F[0:3, 3:7] = -skew_rotated_delta_pos @ (Jacobian of R wrt q) ... (这部分比较复杂)

        # --- 4. 调整过程噪声 Q for delta_time ---
        dt_adjusted = max(abs(delta_time), 1e-6) # 确保时间间隔为正且不为零
        Q = self.Q_per_sec * dt_adjusted # 过程噪声协方差随时间间隔进行缩放

        # --- 5. 预测协方差 ---
        # P_pred = F * P_prev * F^T + Q
        # 当 F = eye(7) 时, P_pred = P_prev + Q
        predicted_covariance = prev_cov + Q # 预测的协方差
        # 确保协方差矩阵的对称性
        predicted_covariance = (predicted_covariance + predicted_covariance.T) / 2.0

        # 缓存预测结果，用于后续可能的平滑处理
        self._last_predicted_state = predicted_state.copy()

        return predicted_state, predicted_covariance # 返回预测的状态和协方差

    def _update(self, predicted_state: np.ndarray, predicted_covariance: np.ndarray,
                gps_pos: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        EKF 更新步骤：使用 GPS 位置测量修正状态估计。

        输入:
            predicted_state (np.ndarray): 从预测步骤得到的预测状态向量 [x,y,z, qx,qy,qz,qw]
            predicted_covariance (np.ndarray): 从预测步骤得到的预测协方差矩阵 (7x7)
            gps_pos (np.ndarray): GPS测量得到的位置 [x, y, z] (UTM坐标)

        输出:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                - updated_state (Optional[np.ndarray]): 更新后的状态向量。如果更新失败则为 None。
                - updated_covariance (Optional[np.ndarray]): 更新后的协方差矩阵。如果更新失败则为 None。
        """
        if gps_pos.shape != (3,) or np.isnan(gps_pos).any(): # 检查GPS测量是否有效
            return None, None # 如果无效（维度错误或包含NaN），则不进行更新

        # 测量矩阵 H (观测模型)：我们只观测位置 x, y, z
        # H 将7维状态向量映射到3维的观测空间 (GPS位置)
        H = np.zeros((3, 7)) # 初始化为零矩阵
        H[0, 0] = 1.0 # 观测x对应状态x
        H[1, 1] = 1.0 # 观测y对应状态y
        H[2, 2] = 1.0 # 观测z对应状态z

        try:
            # 1. 计算测量残差 (Innovation): y = z_gps - h(x_predicted)
            #    由于观测模型是线性的 (h(x) = Hx)，所以 h(x_predicted) = H @ x_predicted
            #    在这里，H @ x_predicted 就是预测状态中的位置部分 predicted_state[:3]
            innovation = gps_pos - predicted_state[:3] # 测量残差

            # 2. 计算残差协方差 (Innovation Covariance): S = H * P_pred * H^T + R
            H_P = H @ predicted_covariance # H * P_pred
            S = H_P @ H.T + self.R # 残差协方差
            S = (S + S.T) / 2.0 # 确保S的对称性

            # 检查 S 是否奇异或接近奇异 (行列式接近0)
            if abs(np.linalg.det(S)) < 1e-12: # 如果S的行列式过小
                # print(f"警告 (EKF Update): S 矩阵接近奇异，跳过更新。 det(S)={np.linalg.det(S)}") # 打印警告
                return None, None # 跳过更新

            # 3. 计算卡尔曼增益 (Kalman Gain): K = P_pred * H^T * S^(-1)
            S_inv = np.linalg.inv(S) # 计算S的逆矩阵
            K = predicted_covariance @ H.T @ S_inv # 卡尔曼增益

            # 4. 状态更新: x_new = x_predicted + K * y
            updated_state = predicted_state + K @ innovation # 更新状态向量
            # 更新后必须重新归一化四元数部分，以保持其单位模长特性
            updated_state[3:] = self.normalize_quaternion(updated_state[3:])

            # 5. 协方差更新: P_new = (I - K * H) * P_predicted (标准形式)
            #    或者使用 Joseph 形式 (更数值稳定): P_new = (I - K@H) @ P_pred @ (I - K@H).T + K @ R @ K.T
            I = np.eye(7) # 7x7单位矩阵
            # 使用 Joseph 形式更新协方差
            P_new = (I - K @ H) @ predicted_covariance @ (I - K @ H).T + K @ self.R @ K.T
            # 也可以用标准形式： P_new = (I - K @ H) @ predicted_covariance
            updated_covariance = (P_new + P_new.T) / 2.0 # 确保更新后的协方差对称

            return updated_state, updated_covariance # 返回更新后的状态和协方差

        except np.linalg.LinAlgError as e: # 捕获更新步骤中的线性代数错误 (例如矩阵求逆失败)
            print(f"警告 (EKF Update): 更新步骤中发生线性代数错误: {e}。跳过更新。") # 打印警告
            return None, None # 跳过更新
        except Exception as e: # 捕获其他未知错误
            print(f"警告 (EKF Update): 更新步骤中发生未知错误: {e}。跳过更新。") # 打印警告
            traceback.print_exc() # 打印详细错误追踪
            return None, None # 跳过更新

    def process_step(self, slam_motion_update: Tuple[np.ndarray, np.ndarray],
                     gps_measurement: Optional[np.ndarray], gnss_available: bool,
                     delta_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理一个时间步：执行EKF的预测、(如果GPS可用则)更新，并应用平滑逻辑。
        该函数会更新EKF的内部状态 (self.state, self.cov)。

        输入:
            slam_motion_update (Tuple[np.ndarray, np.ndarray]): SLAM的相对运动 (delta_pos_local, delta_quat)
            gps_measurement (Optional[np.ndarray]): 当前时间点的GPS位置测量 [x,y,z]。如果GNSS不可用或无有效测量，则为None。
            gnss_available (bool): 当前时间点GNSS是否可用 (即是否有有效的GPS测量用于更新)。
            delta_time (float): 与上一个处理步骤的时间间隔 (秒)。

        输出:
            Tuple[np.ndarray, np.ndarray]:
                - final_fused_pos (np.ndarray): 当前步骤融合后的最佳位置估计 [x,y,z]
                - final_fused_quat (np.ndarray): 当前步骤融合后的最佳姿态估计 [qx,qy,qz,qw]
        """
        # --- 1. EKF 预测 ---
        # 基于SLAM的运动信息，进行状态和协方差的预测
        predicted_state, predicted_covariance = self._predict(slam_motion_update, delta_time)

        # --- 2. EKF 更新 (如果GNSS可用且有测量数据) ---
        updated_state = None # 初始化更新后的状态为None
        updated_covariance = None # 初始化更新后的协方差为None
        if gnss_available and gps_measurement is not None: # 如果GNSS可用且有测量数据
            # 使用预测结果和GPS测量进行更新
            res = self._update(predicted_state, predicted_covariance, gps_measurement)
            if res[0] is not None: # 检查更新是否成功 (返回的状态不是None)
                updated_state, updated_covariance = res # 获取更新后的状态和协方差

        # --- 3. 处理GNSS状态变化和平滑权重 ---
        # 判断GNSS是否刚刚从不可用恢复到可用
        just_recovered = gnss_available and (self.gnss_available_prev == False)

        if gnss_available: # 如果当前GNSS可用
            if just_recovered: # 如果是刚刚恢复
                self.gnss_update_weight = self.weight_delta # 初始化权重为一个小的增量
            elif self.gnss_update_weight < 1.0: # 如果仍在过渡期 (权重小于1)
                # 逐渐增加权重，直到达到1.0
                self.gnss_update_weight = min(1.0, self.gnss_update_weight + self.weight_delta)
            else: # 如果已经过了过渡期
                self.gnss_update_weight = 1.0 # 权重保持为1.0
        else: # 如果当前GNSS不可用
            self.gnss_update_weight = 0.0 # 权重设为0.0 (完全依赖预测)

        # --- 4. 计算最终融合状态 (应用平滑) ---
        final_fused_state = predicted_state # 默认情况下，最终状态是预测状态
        final_fused_covariance = predicted_covariance # 默认情况下，最终协方差是预测协方差

        # 平滑逻辑：仅在GNSS可用、EKF更新成功、且平滑权重小于1.0时进行
        # 目的是在GNSS刚恢复时，平滑地从纯预测状态过渡到融合了GPS的状态
        if gnss_available and updated_state is not None and self.gnss_update_weight < 1.0:
            w = self.gnss_update_weight # 当前GNSS更新结果的权重
            # 用于平滑插值的“预测”状态，应该是没有经过当前GPS更新的纯航位推算结果
            # _last_predicted_state 存储的是上一个 predict 步骤的结果，正好符合需求
            pred_state_for_smooth = self._last_predicted_state # 使用上一步的预测结果作为平滑的起点

            # 位置插值: (1-w)*预测位置 + w*更新后位置
            smooth_pos = (1.0 - w) * pred_state_for_smooth[:3] + w * updated_state[:3]
            # 姿态插值 (使用NLERP): (1-w)*预测姿态 + w*更新后姿态 (归一化)
            smooth_quat = quaternion_nlerp(pred_state_for_smooth[3:], updated_state[3:], w)

            final_fused_state = np.concatenate([smooth_pos, smooth_quat]) # 组合成平滑后的状态
            # 平滑期间的协方差: 简单地使用更新后的协方差，因为它代表了融合GPS信息后的更优估计
            final_fused_covariance = updated_covariance

        # 如果 GNSS 可用，并且EKF更新成功，且平滑已完成 (权重为1.0) 或不需要平滑 (例如一直有GNSS)
        elif gnss_available and updated_state is not None:
             final_fused_state = updated_state # 直接使用EKF更新后的状态
             final_fused_covariance = updated_covariance # 使用EKF更新后的协方差

        # 如果 GNSS 不可用，则最终状态就是预测状态 (已在上面默认设置)

        # --- 5. 更新EKF内部状态，为下一个时间步做准备 ---
        self.state = final_fused_state.copy() # 更新EKF的当前状态
        self.cov = final_fused_covariance.copy() # 更新EKF的当前协方差

        # --- 6. 更新上一步GNSS状态，用于下一次判断是否“刚刚恢复” ---
        self.gnss_available_prev = gnss_available

        # --- 7. 返回当前步最终融合结果 (位置和姿态) ---
        return self.state[:3].copy(), self.state[3:].copy() # 返回位置和四元数

# ----------------------------
# EKF 轨迹修正应用函数
# ----------------------------

def apply_ekf_correction(slam_data: Dict[str, np.ndarray], gps_data: Dict[str, np.ndarray],
                         sim3_pos: np.ndarray, sim3_quat: np.ndarray,
                         config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用 EKF 对经过 Sim3 全局变换的 SLAM 轨迹进行逐点融合修正。
    这个函数会初始化并驱动 EKF 滤波器处理整个轨迹。

    输入:
        slam_data (Dict[str, np.ndarray]): 原始SLAM数据字典，包含 'timestamps', 'positions', 'quaternions'。
        gps_data (Dict[str, np.ndarray]): 处理后的GPS数据字典，包含 'timestamps', 'positions' (UTM), 'projector'。
        sim3_pos (np.ndarray): 经过Sim3全局对齐后的SLAM轨迹位置 (N_slam, 3)。
        sim3_quat (np.ndarray): 经过Sim3全局对齐后的SLAM轨迹姿态四元数 (N_slam, 4)。
        config (Dict[str, Any]): 包含 EKF 和时间对齐配置的总配置字典。

    输出:
        Tuple[np.ndarray, np.ndarray]:
            - corrected_pos_list (np.ndarray): EKF修正后的轨迹位置 (N_slam, 3)。
            - corrected_quat_list (np.ndarray): EKF修正后的轨迹姿态四元数 (N_slam, 4)。
    """
    n_points = len(slam_data['timestamps']) # 获取SLAM轨迹的点数
    if n_points == 0: # 如果SLAM数据为空
        print("错误 (apply_ekf_correction): 输入的 SLAM 数据为空。") # 打印错误
        return np.empty((0, 3)), np.empty((0, 4)) # 返回空数组
    if not (sim3_pos.shape[0] == n_points and sim3_quat.shape[0] == n_points): # 检查Sim3变换后点数是否与原始一致
         raise ValueError(f"错误: Sim3 变换后的轨迹点数与原始 SLAM 时间戳数量不匹配。") # 抛出值错误

    ekf_config = config['ekf'] # 获取EKF相关的配置
    time_align_config = config['time_alignment'] # 获取时间对齐相关的配置

    # 1. 初始化 EKF
    try:
        ekf = ExtendedKalmanFilter( # 创建EKF实例
            initial_pos=sim3_pos[0], # 使用Sim3变换后轨迹的第一个点作为初始位置
            initial_quat=sim3_quat[0], # 使用Sim3变换后轨迹的第一个点作为初始姿态
            config=ekf_config # 传入EKF配置
        )
        # 为了正确初始化EKF的 gnss_available_prev 状态 (判断第一个点是否有GPS)，
        # 需要对齐一次GPS数据到SLAM时间戳。
        aligned_gps_for_init, valid_mask_for_init = dynamic_time_alignment(
            slam_data, gps_data, time_align_config
        )
        # 设置EKF的初始 gnss_available_prev 状态
        ekf.gnss_available_prev = valid_mask_for_init[0] if n_points > 0 and len(valid_mask_for_init) > 0 else False
        print(f"EKF 初始化完成。初始 GNSS 状态 (基于第一个SLAM时间点): {ekf.gnss_available_prev}")

    except Exception as e: # 捕获EKF初始化过程中的错误
         raise ValueError(f"EKF 初始化失败: {e}") # 抛出值错误

    # 2. 获取对齐的 GPS 数据用于 EKF 更新
    # 这里可以直接复用上面为初始化 gnss_available_prev 而进行的对齐结果。
    print("正在使用已对齐的 GPS 数据进行 EKF 更新...")
    aligned_gps_for_update = aligned_gps_for_init # 对齐到SLAM时间戳的GPS位置
    valid_mask_for_update = valid_mask_for_init # 对应的有效性掩码
    num_valid_gps_for_update = np.sum(valid_mask_for_update) # 计算有多少个SLAM时间点可以获得有效的GPS测量
    print(f"EKF 修正：共有 {num_valid_gps_for_update} / {n_points} 个时间点将尝试使用有效的 GPS 测量进行更新。")

    # 准备存储修正后的轨迹数据
    corrected_pos_list = np.zeros_like(sim3_pos) # 初始化修正后的位置列表，形状同sim3_pos
    corrected_quat_list = np.zeros_like(sim3_quat) # 初始化修正后的姿态列表，形状同sim3_quat

    # 第一个点的状态就是EKF的初始状态 (已经在EKF内部通过构造函数设置好了)
    corrected_pos_list[0], corrected_quat_list[0] = ekf.state[:3].copy(), ekf.state[3:].copy()

    # --- 获取原始 SLAM 位姿用于计算相对运动 ---
    # EKF的预测步骤需要的是原始SLAM系统输出的相对运动，而不是Sim3变换后的。
    orig_slam_pos = slam_data['positions'] # 原始SLAM位置
    orig_slam_quat = slam_data['quaternions'] # 原始SLAM姿态

    last_time = slam_data['timestamps'][0] # 初始化上一个时间戳为第一个SLAM时间戳
    predict_steps = 0 # EKF处理的总步数计数器
    updates_attempted = 0 # 尝试进行GPS更新的步数计数器

    # 3. 迭代处理每个 SLAM 时间点 (从第二个点开始，因为第一个点已作为初始状态)
    for i in range(1, n_points): # 从索引1开始遍历
        current_time = slam_data['timestamps'][i] # 当前SLAM时间戳
        delta_t = current_time - last_time # 计算与上一个时间戳的时间差
        if delta_t <= 1e-9: # 如果时间差过小或为负 (时间戳可能非严格单调)
            # 这种情况可能发生在SLAM数据时间戳不精确或有重复时
            # print(f"警告 (EKF loop): 时间戳间隔过小或非单调在索引 {i} (dt={delta_t:.4f}s)。使用小正值。")
            delta_t = 1e-6 # 使用一个非常小的正时间间隔，避免除零或负时间间隔导致的问题

        # a) 计算 SLAM 的相对运动 (Motion Update)
        # 使用 *原始* SLAM 数据计算从上一个姿态到当前姿态的相对运动
        slam_motion_update = calculate_relative_pose(
            orig_slam_pos[i-1], orig_slam_quat[i-1], # 上一个原始SLAM位姿
            orig_slam_pos[i], orig_slam_quat[i]     # 当前原始SLAM位姿
        )

        # b) 获取当前时间点的 GPS 测量和可用性
        gnss_available = valid_mask_for_update[i] # 当前SLAM时间点是否有对应的有效GPS测量
        gps_measurement = aligned_gps_for_update[i] if gnss_available else None # 获取GPS测量值
        if gps_measurement is not None and np.isnan(gps_measurement).any(): # 双重检查，如果插值结果意外为NaN
            gnss_available = False # 视为无效
            gps_measurement = None # 清空测量值

        # c) 调用 EKF 的核心处理步骤 (process_step)
        # process_step 内部会处理预测、更新、平滑，并更新 EKF 的内部状态
        fused_pos, fused_quat = ekf.process_step(
            slam_motion_update=slam_motion_update, # 传入SLAM相对运动
            gps_measurement=gps_measurement,       # 传入GPS测量 (可能为None)
            gnss_available=gnss_available,         # 传入GNSS可用性
            delta_time=delta_t                     # 传入时间间隔
        )
        predict_steps += 1 # EKF处理步数加一
        if gnss_available: # 如果当前尝试了GPS更新
             updates_attempted += 1 # 尝试更新次数加一

        # d) 记录当前 EKF 修正后的状态 (位置和姿态)
        corrected_pos_list[i] = fused_pos
        corrected_quat_list[i] = fused_quat
        last_time = current_time # 更新上一个时间戳，为下一次迭代做准备

    print(f"EKF 修正完成。共执行 {predict_steps} 步处理。") # 打印EKF处理总结
    print(f"  尝试进行 GPS 更新的步数: {updates_attempted} / {predict_steps}") # 打印尝试更新的统计
    # (EKF类内部可以添加更详细的计数器来跟踪成功/失败/跳过的更新次数)

    return corrected_pos_list, corrected_quat_list # 返回修正后的位置和姿态列表

# ----------------------------
# 主流程控制
# ----------------------------

def main_process_gui():
    """
    主处理流程，包含GUI文件选择、数据加载、预处理、对齐、变换、EKF融合、评估和可视化。

    输入:
        无 (通过GUI获取文件路径)

    输出:
        无 (结果会显示或保存到文件)
    """
    slam_path = "" # 初始化SLAM文件路径为空字符串
    gps_path = "" # 初始化GPS文件路径为空字符串
    try:
        # 1. 文件选择
        slam_path = select_slam_file() # 打开对话框让用户选择SLAM文件
        if not slam_path: print("未选择 SLAM 文件，操作取消。"); return # 如果用户未选择，则打印信息并返回
        gps_path = select_gps_file() # 打开对话框让用户选择GPS文件
        if not gps_path: print("未选择 GPS 文件，操作取消。"); return # 如果用户未选择，则打印信息并返回

        print("-" * 30) # 打印分隔线
        print(f"选择的 SLAM 文件: {slam_path}") # 打印选择的SLAM文件路径
        print(f"选择的 GPS 文件: {gps_path}") # 打印选择的GPS文件路径
        print("-" * 30) # 打印分隔线

        # 2. 数据加载与预处理 (包含 GPS RANSAC 过滤)
        print("步骤 1/7: 加载并预处理数据...") # 打印当前步骤信息
        slam_data = load_slam_trajectory(slam_path) # 加载SLAM轨迹数据
        gps_data = load_gps_data(gps_path) # 加载GPS数据 (内部会进行UTM投影和RANSAC过滤)
        print(f"  SLAM 点数: {len(slam_data['positions'])}") # 打印加载的SLAM点数
        print(f"  GPS 点数 (有效、投影并过滤后): {len(gps_data['positions'])}") # 打印处理后的GPS点数
        if len(slam_data['positions']) == 0 or len(gps_data['positions']) < 2: # 检查数据是否足够进行后续处理
             raise ValueError("SLAM 数据为空 或 有效 GPS 数据点不足 (<2)，无法继续处理。") # 如果不足，抛出值错误
        print("数据加载与预处理完成。") # 打印完成信息
        print("-" * 30) # 打印分隔线

        # 3. 时间对齐 (获取用于 Sim3 的匹配点)
        print("步骤 2/7: 执行时间对齐以获取 Sim3 匹配点...") # 打印当前步骤信息
        # 对齐GPS数据到SLAM时间戳，以找到用于Sim3变换估计的对应点
        aligned_gps_for_sim3, valid_mask_sim3 = dynamic_time_alignment(
            slam_data, gps_data, CONFIG['time_alignment'] # 使用配置中的时间对齐参数
        )
        valid_indices_sim3 = np.where(valid_mask_sim3)[0] # 获取那些成功对齐的SLAM时间戳的索引
        min_points_needed_for_sim3 = CONFIG['sim3_ransac']['min_inliers_needed'] # 获取Sim3 RANSAC所需的最小内点数
        if len(valid_indices_sim3) < min_points_needed_for_sim3: # 如果有效匹配点不足
            raise ValueError(f"有效时间同步匹配点不足 ({len(valid_indices_sim3)} < {min_points_needed_for_sim3})，无法进行 Sim3 变换估计。") # 抛出值错误
        print(f"找到 {len(valid_indices_sim3)} 个有效时间同步的匹配点用于 Sim3 估计。") # 打印找到的匹配点数
        print("时间对齐完成。") # 打印完成信息
        print("-" * 30) # 打印分隔线

        # 4. Sim3 全局对齐
        print("步骤 3/7: 计算稳健的 Sim3 全局变换...") # 打印当前步骤信息
        src_points = slam_data['positions'][valid_indices_sim3] # 源点集：原始SLAM轨迹中与GPS对齐上的点
        dst_points = aligned_gps_for_sim3[valid_indices_sim3] # 目标点集：对应的对齐后的GPS位置
        # 使用RANSAC稳健地计算Sim3变换 (旋转R, 平移t, 尺度s)
        R, t, scale = compute_sim3_transform_robust(
            src=src_points, dst=dst_points, **CONFIG['sim3_ransac'] # 使用字典解包传递Sim3 RANSAC配置参数
        )
        if R is None or t is None or scale is None: # 如果Sim3计算失败
             raise RuntimeError("Sim3 全局变换计算失败，无法继续。") # 抛出运行时错误
        print("Sim3 全局变换计算成功。") # 打印成功信息
        print("步骤 4/7: 应用 Sim3 变换到整个 SLAM 轨迹...") # 打印当前步骤信息
        # 将计算得到的Sim3变换应用到整个原始SLAM轨迹 (位置和姿态)
        sim3_pos, sim3_quat = transform_trajectory(
            slam_data['positions'], slam_data['quaternions'], R, t, scale
        )
        print("Sim3 变换应用完成。") # 打印完成信息
        print("-" * 30) # 打印分隔线

        # 5. EKF 局部修正 (融合)
        print("步骤 5/7: 应用 EKF 进行轨迹融合与修正...") # 打印当前步骤信息
        # 调用EKF修正函数，对Sim3变换后的轨迹进行逐点融合
        corrected_pos, corrected_quat = apply_ekf_correction(
            slam_data, gps_data, sim3_pos, sim3_quat, CONFIG # 传入原始SLAM数据、GPS数据、Sim3变换后轨迹和总配置
        )
        print("EKF 轨迹融合与修正完成。") # 打印完成信息
        print("-" * 30) # 打印分隔线

        # 6. 评估误差
        print("步骤 6/7: 评估轨迹误差...") # 打印当前步骤信息
        # 再次进行时间对齐，以获取用于最终误差评估的对齐GPS点
        # (理论上可以复用之前的对齐结果，但为了模块清晰性，这里可能重新计算或确保使用正确的对齐集)
        aligned_gps_for_eval, valid_mask_eval = dynamic_time_alignment(
             slam_data, gps_data, CONFIG['time_alignment']
        )
        valid_indices_eval = np.where(valid_mask_eval)[0] # 获取用于评估的有效SLAM索引
        eval_gps_points = np.empty((0,3)) # 初始化用于评估的GPS点集为空
        ekf_errors = None # 初始化EKF误差为空
        if len(valid_indices_eval) > 0: # 如果有有效的对齐点用于评估
            eval_gps_points = aligned_gps_for_eval[valid_indices_eval] # 获取这些对齐的GPS点
            print(f"  (基于 {len(valid_indices_eval)} 个有效对齐点进行评估)") # 打印评估点数
            # 评估原始SLAM轨迹与对齐GPS的误差
            raw_positions_eval = slam_data['positions'][valid_indices_eval]
            raw_errors = np.linalg.norm(raw_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] 原始轨迹   vs 对齐GPS -> 均值误差: {np.mean(raw_errors):.3f} m, 中位数误差: {np.median(raw_errors):.3f} m")
            # 评估Sim3对齐后轨迹与对齐GPS的误差
            sim3_positions_eval = sim3_pos[valid_indices_eval]
            sim3_errors = np.linalg.norm(sim3_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] Sim3 对齐后 vs 对齐GPS -> 均值误差: {np.mean(sim3_errors):.3f} m, 中位数误差: {np.median(sim3_errors):.3f} m")
            # 评估EKF融合后轨迹与对齐GPS的误差
            ekf_positions_eval = corrected_pos[valid_indices_eval]
            ekf_errors = np.linalg.norm(ekf_positions_eval - eval_gps_points, axis=1)
            print(f"  [评估] EKF 融合后  vs 对齐GPS -> 均值误差: {np.mean(ekf_errors):.3f} m, 中位数误差: {np.median(ekf_errors):.3f} m")
        else: # 如果没有有效的对齐点用于评估
            print("警告：无有效对齐的GPS点用于最终误差评估。") # 打印警告
            ekf_errors = None # EKF误差保持为None
        print("误差评估完成。") # 打印完成信息
        print("-" * 30) # 打印分隔线

        # 7. 保存结果
        print("步骤 7/7: 保存结果与可视化...") # 打印当前步骤信息
        # 弹出对话框询问用户是否保存修正后的轨迹
        save_results = messagebox.askyesno("保存结果", "处理完成。是否要保存修正后的轨迹?")
        if save_results: # 如果用户选择是
            # 从SLAM文件路径中提取基本文件名，用于生成默认保存文件名
            base_filename = slam_path.split('/')[-1].split('\\')[-1]
            default_save_name = base_filename.replace('.txt', '_corrected_utm.txt')
            if default_save_name == base_filename: default_save_name += '_corrected_utm.txt' # 避免文件名不变

            # 打开保存文件对话框，让用户选择保存UTM坐标轨迹的路径和文件名
            output_path_utm = filedialog.asksaveasfilename(
                title="保存修正后的轨迹 (UTM 坐标, TUM 格式)", defaultextension=".txt",
                filetypes=[("文本文件", "*.txt")], initialfile=default_save_name
            )
            if output_path_utm: # 如果用户选择了保存路径
                # 准备要保存的数据：时间戳, 修正后的UTM位置, 修正后的姿态四元数
                output_data_utm = np.column_stack((slam_data['timestamps'], corrected_pos, corrected_quat))
                # 保存为文本文件，指定格式
                np.savetxt(output_path_utm, output_data_utm,
                           fmt=['%.6f'] + ['%.6f'] * 3 + ['%.8f'] * 4, # 时间戳6位小数，位置6位小数，四元数8位小数
                           header="timestamp x y z qx qy qz qw (UTM)", comments='') # 文件头信息
                print(f"  UTM 坐标轨迹已保存至: {output_path_utm}") # 打印保存成功信息
                # --- 尝试保存 WGS84 格式的轨迹 ---
                try:
                    projector = gps_data.get('projector') # 获取GPS数据加载时创建的投影对象
                    if projector: # 如果投影对象存在
                        print("  正在将修正后的轨迹转换回 WGS84 坐标...") # 打印转换信息
                        # 将修正后的UTM位置转换回WGS84经纬度高程
                        wgs84_lon_lat_alt = utm_to_wgs84(corrected_pos, projector)
                        # 准备WGS84格式的数据：时间戳, 经度, 纬度, 高程, 修正后的姿态四元数
                        output_data_wgs84 = np.column_stack((
                            slam_data['timestamps'], wgs84_lon_lat_alt[:, 0], wgs84_lon_lat_alt[:, 1], wgs84_lon_lat_alt[:, 2], corrected_quat
                        ))
                        # 生成WGS84格式的默认保存文件名
                        output_path_wgs84 = output_path_utm.replace('_utm.txt', '_wgs84.txt')
                        if output_path_wgs84 == output_path_utm: output_path_wgs84 = output_path_utm.replace('.txt', '_wgs84.txt')
                        # 保存WGS84格式的轨迹
                        np.savetxt(output_path_wgs84, output_data_wgs84,
                                   fmt=['%.6f'] + ['%.8f', '%.8f', '%.3f'] + ['%.8f'] * 4, # 时间戳6位，经纬度8位，高程3位，四元数8位
                                   header="timestamp lon lat alt qx qy qz qw (WGS84)", comments='') # 文件头
                        print(f"  WGS84 坐标轨迹已保存至: {output_path_wgs84}") # 打印保存成功信息
                    else: print("警告：GPS 数据中缺少投影仪对象，无法保存 WGS84 格式。") # 如果没有投影对象，打印警告
                except Exception as e: print(f"错误：保存 WGS84 格式轨迹失败: {str(e)}") # 捕获保存WGS84时的错误
            else: print("  用户取消保存。") # 如果用户取消保存，打印信息

        # 8. 可视化
        print("  正在生成可视化结果图...") # 打印信息
        plot_results( # 调用绘图函数
            original_pos=slam_data['positions'], # 原始SLAM位置
            corrected_pos=corrected_pos, # EKF修正后位置
            gps_pos=gps_data['positions'], # 过滤后的GPS UTM点
            valid_indices_for_error=valid_indices_eval, # 用于误差评估的有效索引
            aligned_gps_for_error=eval_gps_points, # 对齐到SLAM时间的GPS点 (用于误差评估)
            slam_times=slam_data['timestamps'], # SLAM时间戳
            ekf_errors=ekf_errors # 计算得到的EKF误差
        )

        print("-" * 30) # 打印分隔线
        print("所有处理步骤完成。") # 打印完成信息
        messagebox.showinfo("完成", "轨迹对齐与融合处理完成！") # 弹出消息框提示用户处理完成

    # --- 异常处理块 ---
    except ValueError as ve: # 捕获值错误
        error_msg = f"处理失败 (值错误): {str(ve)}" # 构建错误消息
        print(f"\n错误!\n{error_msg}") # 打印错误信息到控制台
        # 附加文件路径信息到错误消息，方便调试
        if gps_path and not slam_path: error_msg += f"\nGPS 文件: {gps_path}"
        if slam_path and not gps_path: error_msg += f"\nSLAM 文件: {slam_path}"
        if slam_path and gps_path: error_msg += f"\nSLAM 文件: {slam_path}\nGPS 文件: {gps_path}"
        traceback.print_exc(); messagebox.showerror("处理失败", error_msg) # 打印详细追踪并显示错误消息框
    except FileNotFoundError as fnf: # 捕获文件未找到错误
        error_msg = f"处理失败 (文件未找到): {str(fnf)}"
        print(f"\n错误!\n{error_msg}"); messagebox.showerror("处理失败", error_msg)
    except AssertionError as ae: # 捕获断言错误 (通常是数据格式问题)
        error_msg = f"处理失败 (数据格式断言错误): {str(ae)}"
        print(f"\n错误!\n{error_msg}")
        if gps_path and not slam_path: error_msg += f"\nGPS 文件: {gps_path}"
        if slam_path and not gps_path: error_msg += f"\nSLAM 文件: {slam_path}"
        if slam_path and gps_path: error_msg += f"\nSLAM 文件: {slam_path}\nGPS 文件: {gps_path}"
        traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except CRSError as ce: # 捕获坐标参考系统错误 (来自pyproj)
         error_msg = f"处理失败 (坐标投影错误): {str(ce)}"
         print(f"\n错误!\n{error_msg}")
         traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except np.linalg.LinAlgError as lae: # 捕获线性代数计算错误 (例如矩阵奇异)
         error_msg = f"处理失败 (线性代数计算错误): {str(lae)}"
         print(f"\n错误!\n{error_msg}")
         traceback.print_exc(); messagebox.showerror("处理失败", error_msg)
    except RuntimeError as rte: # 捕获由Sim3失败等引发的运行时错误
         error_msg = f"处理失败 (运行时错误): {str(rte)}"
         print(f"\n错误!\n{error_msg}")
         messagebox.showerror("处理失败", error_msg)
    except Exception as e: # 捕获所有其他未预料的异常
        error_msg = f"处理过程中发生未预料的错误: {type(e).__name__}: {str(e)}" # 构建通用错误消息
        print(f"\n严重错误!\n{error_msg}") # 打印错误信息
        if gps_path and not slam_path: error_msg += f"\nGPS 文件: {gps_path}"
        if slam_path and not gps_path: error_msg += f"\nSLAM 文件: {slam_path}"
        if slam_path and gps_path: error_msg += f"\nSLAM 文件: {slam_path}\nGPS 文件: {gps_path}"
        traceback.print_exc() # 打印详细的异常追踪信息
        messagebox.showerror("处理失败", f"{error_msg}\n\n详情请查看控制台输出。") # 显示错误消息框

if __name__ == "__main__": # 如果脚本作为主程序运行
    print("启动 SLAM-GPS 轨迹对齐与融合工具 (实现DR+平滑恢复)...") # 打印启动信息
    print("="*70) # 打印分隔线
    print("配置参数概览:") # 打印配置参数概览标题
    # 打印GPS RANSAC滤波相关的配置参数
    print(f"  GPS RANSAC 滤波启用: {CONFIG['gps_filtering_ransac']['enabled']}")
    if CONFIG['gps_filtering_ransac']['enabled']: # 如果启用了GPS RANSAC
        use_sw = CONFIG['gps_filtering_ransac']['use_sliding_window'] # 获取是否使用滑动窗口
        print(f"  GPS RANSAC 模式: {'滑动窗口' if use_sw else '全局'}") # 打印RANSAC模式
        if use_sw: # 如果使用滑动窗口
            print(f"    窗口时长: {CONFIG['gps_filtering_ransac']['window_duration_seconds']} s") # 打印窗口时长
            print(f"    窗口步长因子: {CONFIG['gps_filtering_ransac']['window_step_factor']}") # 打印窗口步长因子
        print(f"    多项式阶数: {CONFIG['gps_filtering_ransac']['polynomial_degree']}") # 打印多项式阶数
        print(f"    最小样本数: {CONFIG['gps_filtering_ransac']['min_samples']}") # 打印最小样本数
        print(f"    残差阈值: {CONFIG['gps_filtering_ransac']['residual_threshold_meters']} m") # 打印残差阈值
        print(f"    最大迭代次数: {CONFIG['gps_filtering_ransac']['max_trials']}") # 打印最大迭代次数
    print(f"  GPS 中断阈值: {CONFIG['time_alignment']['max_gps_gap_threshold']} s") # 打印GPS中断阈值
    print(f"  Sim3 RANSAC 最小内点数: {CONFIG['sim3_ransac']['min_inliers_needed']}") # 打印Sim3 RANSAC最小内点数
    print(f"  EKF GNSS 恢复平滑步数: {CONFIG['ekf']['transition_steps']}") # 打印EKF GNSS恢复平滑步数
    print("="*70) # 打印分隔线

    main_process_gui() # 调用主处理函数，启动程序
    print("\n程序结束。") # 打印程序结束信息
