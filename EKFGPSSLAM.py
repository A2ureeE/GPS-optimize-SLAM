import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from pyproj import Proj
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog

# ----------------------------
# 增强数据加载与预处理
# ----------------------------

def load_slam_trajectory(txt_path):
    """加载并验证SLAM轨迹数据"""
    try:
        data = np.loadtxt(txt_path)
        #文件输入需要TUM格式：Time Stamp, Position X, Position Y, Position Z, Quaternion X, Quaternion Y, Quaternion Z, Quaternion W
        assert data.shape[1] == 8, "SLAM文件格式错误：需要8列数据"
        return {
            'timestamps': data[:, 0],
            'positions': data[:, 1:4],
            'quaternions': data[:, 4:8]
        }
    except Exception as e:
        raise ValueError(f"SLAM数据加载失败: {str(e)}")

def auto_utm_projection(lons, lats):
    """根据经度自动计算UTM分区"""
    central_lon = np.mean(lons)
    zone = int((central_lon + 180) // 6 + 1)
    hemisphere = '' if np.mean(lats) >= 0 else ' +south'
    return f"{zone}{hemisphere}"


def load_gps_data(txt_path):
    """加载GPS数据时保存UTM投影参数"""
    try:
        gps_data = np.loadtxt(txt_path, delimiter=' ')
        assert gps_data.shape[1] >= 4, "GPS文件需要至少4列数据（时间戳、纬度、经度、海拔）"

        lons = gps_data[:, 2]
        lats = gps_data[:, 1]
        utm_zone = auto_utm_projection(lons, lats)

        # 创建投影并保存关键参数
        proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84')
        x, y = proj(lons, lats)

        return {
            'timestamps': gps_data[:, 0],
            'positions': np.column_stack((x, y, gps_data[:, 3])),
            'utm_zone': utm_zone,  # 存储UTM分区
            'projector': proj  # 存储投影对象
        }
    except Exception as e:
        raise ValueError(f"GPS数据加载失败: {str(e)}")


def utm_to_wgs84(utm_points, projector):
    """将UTM坐标批量转换为WGS84经纬度
    参数：
    - utm_points: Nx3数组，格式为[X, Y, Z]
    - projector: pyproj.Proj实例（必须与原始投影参数一致）

    返回：
    - Nx3数组，格式为[经度, 纬度, 高程]
    """
    lons, lats = projector(utm_points[:, 0], utm_points[:, 1], inverse=True)
    return np.column_stack((lons, lats, utm_points[:, 2]))
# ----------------------------
# 时间对齐与变换计算（保持不变）
# ----------------------------
def estimate_time_offset(slam_times, gps_times):
    """通过互相关估计时钟偏移"""
    max_samples = min(500, len(slam_times), len(gps_times))
    slam_sample = np.linspace(slam_times.min(), slam_times.max(), max_samples)
    gps_sample = np.linspace(gps_times.min(), gps_times.max(), max_samples)

    # 归一化处理
    slam_norm = (slam_sample - np.mean(slam_sample)) / np.std(slam_sample)
    gps_norm = (gps_sample - np.mean(gps_sample)) / np.std(gps_sample)

    # 计算互相关
    corr = np.correlate(slam_norm, gps_norm, mode='full')
    peak_idx = corr.argmax()
    offset = (peak_idx - len(slam_norm) + 1) * (slam_times[1] - slam_times[0])

    return offset


def dynamic_time_alignment(slam_data, gps_data):
    """动态时间同步系统"""
    # 粗粒度偏移估计
    offset = estimate_time_offset(slam_data['timestamps'], gps_data['timestamps'])
    adjusted_gps_times = gps_data['timestamps'] + offset

    # 精确对齐
    valid_mask = (
            (slam_data['timestamps'] >= adjusted_gps_times.min()) &
            (slam_data['timestamps'] <= adjusted_gps_times.max())
    )

    # 分段三次样条插值
    interp_func = interp1d(
        adjusted_gps_times,
        gps_data['positions'],
        axis=0,
        kind='cubic',
        bounds_error=False,
        fill_value=np.nan
    )

    aligned_gps = interp_func(slam_data['timestamps'])
    return aligned_gps, valid_mask


# ----------------------------
# 3. 鲁棒变换估计模块
# ----------------------------

def compute_sim3_transform_robust(src, dst):
    """带RANSAC的Sim3变换估计"""
    # 数据标准化
    src_mean, src_std = np.mean(src, axis=0), np.std(src, axis=0)
    dst_mean, dst_std = np.mean(dst, axis=0), np.std(dst, axis=0)

    src_norm = (src - src_mean) / src_std
    dst_norm = (dst - dst_mean) / dst_std
#打印这两个值

    # RANSAC筛选
    ransac = RANSACRegressor(min_samples=4, residual_threshold=0.1,max_trials=1000)
    ransac.fit(src_norm, dst_norm)
    inlier_mask = ransac.inlier_mask_
    # 仅在足够内点时计算
    if np.sum(inlier_mask) < 3:
        raise ValueError("有效对应点不足，无法计算可靠变换")

    return compute_sim3_transform(src[inlier_mask], dst[inlier_mask])


def compute_sim3_transform(src, dst):
    src_centroid = np.mean(src, axis=0)
    dst_centroid = np.mean(dst, axis=0)

    H = (src - src_centroid).T @ (dst - dst_centroid)
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        R = Vt.T @ np.diag([1, 1, -1]) @ U.T

    # 计算每对点到质心的距离比例，取中位数
    src_dist = np.linalg.norm(src - src_centroid, axis=1)
    dst_dist = np.linalg.norm(dst - dst_centroid, axis=1)
    valid_ratios = dst_dist / (src_dist + 1e-9)  # 防止除以零
    scale = np.median(valid_ratios[valid_ratios > 0])  # 过滤无效值

    t = dst_centroid - scale * R @ src_centroid
    return R, t, scale


# ----------------------------
# 4. 轨迹变换与姿态更新
# ----------------------------

def transform_trajectory(positions, quaternions, R, t, scale):
    """全姿态变换（位置+旋转）"""
    # 位置变换
    trans_pos = scale * (R @ positions.T).T + t

    # 旋转变换
    R_rot = Rotation.from_matrix(R)
    quat_array = []
    for q in quaternions:
        original_rot = Rotation.from_quat(q[[1, 2, 3, 0]])  # 输入顺序: qx,qy,qz,qw
        new_rot = R_rot * original_rot
        quat_array.append(new_rot.as_quat()[[3, 0, 1, 2]])  # 输出顺序: qw,qx,qy,qz

    return trans_pos, np.array(quat_array)


# ----------------------------
# 5. 可视化与评估系统
# ----------------------------

def plot_results(original, corrected, gps, errors):
    """增强可视化系统"""
    plt.figure(figsize=(15, 10))

    # 2D轨迹对比
    plt.subplot(2, 2, 1)
    plt.plot(original[:, 0], original[:, 1], 'b-', alpha=0.3, label='Original Trajectory')
    plt.plot(corrected[:, 0], corrected[:, 1], 'g-', label='Corrected Trajectory')

    # 生成颜色数组：第一个蓝、最后一个绿、其他红
    n_gps = len(gps)
    colors = ['r'] * n_gps
    if n_gps >= 1:
        colors[0] = 'b'
    if n_gps >= 2:
        colors[-1] = 'g'

    plt.scatter(gps[:, 0], gps[:, 1], c=colors, marker='*', label='GPS Reference Points')
    plt.title('2DTrace')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)

    # 3D轨迹对比
    ax = plt.subplot(2, 2, 2, projection='3d')
    ax.plot(original[:, 0], original[:, 1], original[:, 2], 'b-', alpha=0.3)
    ax.plot(corrected[:, 0], corrected[:, 1], corrected[:, 2], 'g-')
    ax.scatter(gps[:, 0], gps[:, 1], gps[:, 2], c='r', marker='*')
    ax.set_title('3DTrach')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # 误差分布
    plt.subplot(2, 2, 3)
    plt.hist(errors, bins=20, alpha=0.7, color='purple')
    plt.title(f'Error Distribution\nMean Error: {np.mean(errors):.2f}m ± {np.std(errors):.2f}m')
    plt.xlabel('Error (m)')
    plt.ylabel('Frequency')

    # 误差热力图
    plt.subplot(2, 2, 4)
    distances = np.linalg.norm(corrected - original, axis=1)
    plt.scatter(corrected[:, 0], corrected[:, 1], c=distances, cmap='viridis', s=10)
    plt.colorbar(label='Position Drift (m)')
    plt.title('Local Drift Heatmap')
    plt.tight_layout()
    plt.show()


# ----------------------------
# GUI文件选择功能
# ----------------------------

def select_file_dialog(title, filetypes):
    """通用文件选择对话框"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    root.destroy()  # 关闭Tk实例
    return file_path

def select_slam_file():
    """选择SLAM轨迹文件"""
    return select_file_dialog(
        "选择SLAM轨迹文件",
        [("文本文件", "*.txt"), ("所有文件", "*.*")]
    )

def select_gps_file():
    """选择GPS数据文件"""
    return select_file_dialog(
        "选择GPS数据文件",
        [ ("文本文件", "*.txt"), ("CSV文件", "*.csv"),("所有文件", "*.*")]
    )


# ----------------------------
# 新增局部修正模块
# ----------------------------

def apply_local_correction(slam_data, gps_data, trans_pos, trans_quat,
                           error_threshold=2.0, time_window=5.0):
    """
    鲁棒局部修正模块
    参数：
    - error_threshold: 触发修正的误差阈值（米）
    - time_window: 姿态修正使用的时间窗口（秒）
    """
    # 时间对齐
    aligned_gps, valid_mask = dynamic_time_alignment(slam_data, gps_data)
    valid_indices = np.where(valid_mask & ~np.isnan(aligned_gps).any(axis=1))[0]

    # 计算误差
    aligned_errors = np.linalg.norm(trans_pos[valid_indices] - aligned_gps[valid_indices], axis=1)
    large_error_mask = (aligned_errors > error_threshold)
    large_error_indices = valid_indices[large_error_mask]

    if len(large_error_indices) == 0:
        return trans_pos, trans_quat

    print(f"修正{len(large_error_indices)}个异常点（阈值={error_threshold}m）")

    # 深度拷贝防止污染原始数据
    corrected_pos = trans_pos.copy()
    corrected_quat = trans_quat.copy()

    # 预计算时间序列
    timestamps = slam_data['timestamps']

    for idx in large_error_indices:
        # 位置强制对齐
        corrected_pos[idx] = aligned_gps[idx]

        try:
            # 时间窗口查询
            t = timestamps[idx]
            window_mask = (timestamps >= t - time_window) & (timestamps <= t + time_window)
            window_indices = np.where(window_mask)[0]

            # 排除当前异常点和无效GPS点
            valid_window_indices = [i for i in window_indices
                                    if i not in large_error_indices
                                    and not np.isnan(aligned_gps[i]).any()]

            if len(valid_window_indices) < 3:
                raise ValueError("有效邻近点不足")

            # 计算局部变换
            R, _, _ = compute_sim3_transform(
                slam_data['positions'][valid_window_indices],
                corrected_pos[valid_window_indices]
            )

            # 旋转变换补偿
            original_rot = Rotation.from_quat(corrected_quat[idx][[1, 2, 3, 0]])
            corrected_rot = Rotation.from_matrix(R) * original_rot
            corrected_quat[idx] = corrected_rot.as_quat()[[3, 0, 1, 2]]

        except Exception as e:
            # 退化到线性插值
            prev_idx = find_nearest_valid(timestamps, idx, -1, large_error_indices)
            next_idx = find_nearest_valid(timestamps, idx, 1, large_error_indices)

            if prev_idx is not None and next_idx is not None:
                alpha = (t - timestamps[prev_idx]) / (timestamps[next_idx] - timestamps[prev_idx])
                interp_quat = Rotation.from_quat(corrected_quat[prev_idx]).slerp(
                    Rotation.from_quat(corrected_quat[next_idx]), alpha)
                corrected_quat[idx] = interp_quat.as_quat()[[3, 0, 1, 2]]
            elif prev_idx is not None:
                corrected_quat[idx] = corrected_quat[prev_idx]
            elif next_idx is not None:
                corrected_quat[idx] = corrected_quat[next_idx]

    return corrected_pos, corrected_quat


def find_nearest_valid(timestamps, current_idx, direction, exclude_indices):
    """查找最近的有效点（direction=1向前，-1向后）"""
    step = 1 if direction == 1 else -1
    idx = current_idx + step
    while 0 <= idx < len(timestamps):
        if idx not in exclude_indices:
            return idx
        idx += step
    return None


class ExtendedKalmanFilter:
    def __init__(self, initial_pos, initial_quat, initial_cov, process_noise, meas_noise):
        self.state = np.concatenate([initial_pos, initial_quat])
        self.cov = initial_cov  # 7x7协方差矩阵
        self.Q = process_noise  # 过程噪声
        self.R = meas_noise  # 观测噪声（越小越信任GPS）

    def predict(self, slam_pos, slam_quat, delta_time):
        """使用SLAM输出作为预测"""
        # 状态转移模型：直接采用SLAM输出
        self.state[:3] = slam_pos
        self.state[3:] = slam_quat

        # 协方差传播：增加过程噪声
        self.cov += self.Q * delta_time

    def update(self, gps_pos, current_time):
        """有GPS时更新并记录时间"""
        H = np.zeros((3, 7))
        H[:3, :3] = np.eye(3)

        # 计算卡尔曼增益（修正维度）
        S = H @ self.cov @ H.T + self.R
        K = self.cov @ H.T @ np.linalg.inv(S)  # K形状应为(7,3)

        # 状态更新：仅更新位置部分
        innovation = gps_pos - self.state[:3]
        self.state[:3] += (K[:3] @ innovation)  # 关键修改：取K的前3行

        # 协方差更新保持不变
        self.cov = (np.eye(7) - K @ H) @ self.cov

        self.last_valid_gps_time = current_time


def apply_ekf_correction(slam_data, gps_data, sim3_pos, sim3_quat):
    """EKF局部修正主函数"""
    # 初始化EKF
    initial_cov = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01])#系统状态的初始不确定性
    process_noise = np.diag([0.5, 0.5, 0.5, 0.05, 0.05, 0.05, 0.05])#过程噪声矩阵描述了系统模型中因外部不可预测因素（如模型不准确、外部干扰等）造成的噪声
    meas_noise = np.diag([0.3, 0.3, 0.3])#测量传感器的噪声水平。每个值表示对应传感器测量值的不确定性

    ekf = ExtendedKalmanFilter(
        sim3_pos[0],
        sim3_quat[0],
        initial_cov,
        process_noise,
        meas_noise
    )

    # 时间对齐
    aligned_gps, valid_mask = dynamic_time_alignment(slam_data, gps_data)
    valid_indices = np.where(valid_mask & ~np.isnan(aligned_gps).any(axis=1))[0]

    corrected_pos = []
    corrected_quat = []
    last_time = slam_data['timestamps'][0]

    for i in range(len(slam_data['timestamps'])):
        # 预测步骤：使用SIM3变换后的SLAM数据
        delta_t = slam_data['timestamps'][i] - last_time
        ekf.predict(sim3_pos[i], sim3_quat[i], delta_t)

        # 如果有GPS观测则更新
        if i in valid_indices:
            ekf.update(aligned_gps[i], slam_data['timestamps'][i])

        # 记录状态
        corrected_pos.append(ekf.state[:3].copy())
        corrected_quat.append(ekf.state[3:].copy())
        last_time = slam_data['timestamps'][i]

    return np.array(corrected_pos), np.array(corrected_quat)


# ----------------------------
# 主流程控制
# ----------------------------

def main_process_gui():
    try:
        # 文件选择
        slam_path = select_slam_file()
        gps_path = select_gps_file()
        if not slam_path or not gps_path: return

        # 数据加载
        slam_data = load_slam_trajectory(slam_path)
        gps_data = load_gps_data(gps_path)
        print(f"SLAM点数: {len(slam_data['positions'])}, GPS点数: {len(gps_data['positions'])}")

        # SIM3全局对齐
        aligned_gps, valid_mask = dynamic_time_alignment(slam_data, gps_data)
        valid_indices = np.where(valid_mask & ~np.isnan(aligned_gps).any(axis=1))[0]

        if len(valid_indices) < 4:
            raise ValueError("有效匹配点不足")

        R, t, scale = compute_sim3_transform_robust(
            slam_data['positions'][valid_indices],
            aligned_gps[valid_indices]
        )
        sim3_pos, sim3_quat = transform_trajectory(
            slam_data['positions'],
            slam_data['quaternions'],
            R, t, scale
        )

        # EKF局部修正
        corrected_pos, corrected_quat = apply_ekf_correction(
            slam_data, gps_data, sim3_pos, sim3_quat
        )
        # 对齐后的 GPS（已插值）
        aligned_gps, valid_mask = dynamic_time_alignment(slam_data, gps_data)
        valid_indices = np.where(valid_mask & ~np.isnan(aligned_gps).any(axis=1))[0]

        # 1. 原始轨迹误差
        raw_errors = np.linalg.norm(slam_data['positions'][valid_indices] - aligned_gps[valid_indices], axis=1)
        print(f"[原始轨迹] 均值误差: {np.mean(raw_errors):.2f} m，最大误差: {np.max(raw_errors):.2f} m")

        # 2. SIM3后轨迹误差
        sim3_errors = np.linalg.norm(sim3_pos[valid_indices] - aligned_gps[valid_indices], axis=1)
        print(f"[SIM3对齐后] 均值误差: {np.mean(sim3_errors):.2f} m，最大误差: {np.max(sim3_errors):.2f} m")

        # 3. EKF融合后轨迹误差
        ekf_errors = np.linalg.norm(corrected_pos[valid_indices] - aligned_gps[valid_indices], axis=1)
        print(f"[EKF融合后] 均值误差: {np.mean(ekf_errors):.2f} m，最大误差: {np.max(ekf_errors):.2f} m")

        # 评估误差
        aligned_errors = np.linalg.norm(corrected_pos[valid_indices] - aligned_gps[valid_indices], axis=1)
        print(f"最终误差：均值={np.mean(aligned_errors):.2f}m, 最大={np.max(aligned_errors):.2f}m")

        # 保存结果
        output_data = np.column_stack((
            slam_data['timestamps'],
            corrected_pos,
            corrected_quat
        ))
        output_path = filedialog.asksaveasfilename(
            title="保存校正轨迹",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt")]
        )
        if output_path:
            np.savetxt(output_path, output_data,
                       fmt='%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f',
                       header="timestamp x y z qx qy qz qw")
            print(f"轨迹已保存至：{output_path}")

            # 新增：保存WGS84坐标轨迹
            try:
                # 获取投影对象
                projector = gps_data['projector']
                # UTM转WGS84
                wgs84_xyz = utm_to_wgs84(corrected_pos, projector)
                output_data_wgs84 = np.column_stack((
                    slam_data['timestamps'],
                    wgs84_xyz,
                    corrected_quat
                ))
                output_path_wgs84 = output_path.replace('.txt', '_wgs84.txt')
                np.savetxt(output_path_wgs84, output_data_wgs84,
                           fmt='%.6f %.8f %.8f %.3f %.6f %.6f %.6f %.6f',
                           header="timestamp lon lat alt qx qy qz qw")
                print(f"WGS84轨迹已保存至：{output_path_wgs84}")
            except Exception as e:
                print(f"WGS84轨迹保存失败: {str(e)}")

        # 可视化
        plot_results(slam_data['positions'], corrected_pos, gps_data['positions'], aligned_errors)

    except Exception as e:
        print(f"处理失败：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 直接启动GUI流程
    main_process_gui()