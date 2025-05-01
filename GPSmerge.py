import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from datetime import datetime


def load_timestamps(timestamp_path, time_offset):
    """加载时间戳文件，并调整时间戳"""
    timestamps = []
    original_timestamps = []  # 存储原始时间戳（秒）

    with open(timestamp_path, 'r') as f:
        for line in f:
            timestamp_str = line.strip()
            # 如果时间戳精度超过6位，截取前6位微秒部分
            timestamp_str = timestamp_str[:26]  # 只保留前6位微秒（示例：2011-09-30 11:50:40.354663）
            # 将时间戳字符串转换为 datetime 对象
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            # 将 datetime 对象转换为秒数（UNIX时间戳，1970年1月1日以来的秒数）
            timestamp_seconds = (timestamp - datetime(1970, 1, 1)).total_seconds()
            original_timestamps.append(timestamp_seconds)

    # 第一个时间戳设为用户输入的时间差
    first_timestamp = time_offset
    timestamps.append(first_timestamp)  # 以 float 类型存储第一个时间戳

    # 后续时间戳：每个时间戳是基于与上一个时间戳的差值，再加上用户输入的时间差
    for idx in range(1, len(original_timestamps)):
        # 计算当前时间戳与前一个时间戳的差值
        time_diff = original_timestamps[idx] - original_timestamps[idx - 1]
        # 加上用户输入的时间差
        timestamp_seconds = timestamps[idx - 1] + time_diff + time_offset
        timestamps.append(timestamp_seconds)  # 确保时间戳是 float 类型

    # 将时间戳转换为科学计数法的字符串格式
    formatted_timestamps = [f"{ts:.18e}" for ts in timestamps]
    return formatted_timestamps


def load_data_from_file(data_path):
    """加载数据文件，返回前三列以及质量相关的参数"""
    data = np.loadtxt(data_path)  # 使用numpy加载数据文件
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)  # 如果数据是1维的，转换为二维
    numsats = int(data[0, 25])  # 假设numsats在第26列（索引25）
    velmode = int(data[0, 27])  # 假设velmode在第28列（索引27）
    return data[:, :3], numsats, velmode  # 返回前三列数据，以及numsats和velmode


def create_combined_file(timestamps, data_folder, output_file):
    """生成合并的txt文件"""
    with open(output_file, 'w') as output_f:
        for idx, timestamp in enumerate(timestamps):
            data_file = os.path.join(data_folder, f"{idx:010d}.txt")  # 假设文件命名规则是0000000000.txt等
            if os.path.exists(data_file):
                data, numsats, velmode = load_data_from_file(data_file)
                # 合并时间戳、数据、numsats和velmode
                for row in data:
                    output_f.write(f"{timestamp} {' '.join(map(str, row))} {numsats} {velmode}\n")
            else:
                print(f"警告：找不到文件 {data_file}")


def select_oxts_folder():
    """打开文件夹选择对话框，选择oxts文件夹"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title="选择oxts文件夹")
    return folder_path


def get_time_offset():
    """获取用户输入的时间差"""
    while True:
        try:
            time_offset = float(input("请输入时间差（SLAM序列与GPS序列的时间差，单位为秒）："))
            return time_offset
        except ValueError:
            print("无效输入，请输入一个有效的数字。")


if __name__ == "__main__":
    # 获取时间差
    time_offset = get_time_offset()

    # 选择oxts文件夹
    oxts_folder = select_oxts_folder()

    if not oxts_folder:
        print("未选择文件夹，程序退出。")
        exit()

    # 设置文件路径
    timestamps_file = os.path.join(oxts_folder, 'timestamps.txt')  # 时间戳文件路径
    data_folder = os.path.join(oxts_folder, 'data')  # 数据文件夹路径
    output_file = 'combined_output.txt'  # 输出文件

    # 检查文件和文件夹是否存在
    if not os.path.exists(timestamps_file):
        print(f"找不到时间戳文件：{timestamps_file}")
        exit()

    if not os.path.exists(data_folder):
        print(f"找不到数据文件夹：{data_folder}")
        exit()

    # 加载时间戳并调整
    timestamps = load_timestamps(timestamps_file, time_offset)

    # 创建合并后的文件
    create_combined_file(timestamps, data_folder, output_file)

    print(f"合并文件已保存至：{output_file}")
