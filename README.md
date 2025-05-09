# GPS-optimize-SLAM
环境python3.7

运行方法：python EKFGPSSLAM.py

**GPSmerge.py**将Kitti数据集的oxts文件夹中的GNSS数据提取出来。需要预先输入的量：GNSS时间轴文件起始时间与Kitti图像时间轴起始时间之差

**kitti2tum.py**为官方提供的将kitti的SLAM轨迹转换为TUM格式

**EKFGPSSLAM.py**需要输入TUM格式SLAM文件、GNSS文件格式为**时间戳、lon、lan、elevation**，也就是GPSmerge文件生成的GPS轨迹

# 简介
毕业设计要做的一个部分，通过Sim3处理slam数据，用ekf局部处理slam数据 很简单的结构
