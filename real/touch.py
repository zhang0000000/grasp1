#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2,os,sys

# 获取当前文件（calibrate.py）所在的目录（即 real 目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录（real 的父目录，即 GRCNN 目录，也就是 UR_Robot.py 所在的目录）
parent_dir = os.path.dirname(current_dir)
# 将上级目录添加到 Python 搜索路径
sys.path.append(parent_dir)

from realsenseD415 import Camera
# from realsenseD435 import Camera
from UR_Robot import UR_Robot


# User options (change me)
# --------------- Setup options ---------------
tcp_host_ip = '192.168.5.10' # IP and port to robot arm as TCP client (UR5)
tcp_port = 30003
tool_orientation = [-np.pi,0,0]
# ---------------------------------------------

# Move robot to home pose
robot = UR_Robot(tcp_host_ip,tcp_port)
# robot.move_j([-np.pi, -np.pi/2, np.pi/2, 0, np.pi/2, np.pi])

# grasp_home = [0.4, 0, 0.4, -np.pi, 0, 0]  # change me!
grasp_home = [-0.4, 0.4, 0.3, -np.pi, 0, 0]  # change me!

robot.move_j_p(grasp_home)
# robot.open_gripper()

# Slow down robot
robot.joint_acc = 0.5
robot.joint_vel = 0.5

# Callback function for clicking on OpenCV window
click_point_pix = ()
camera_color_img, camera_depth_img = robot.get_camera_data()
def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:      # 检测左键点击事件
        global camera, robot, click_point_pix
        click_point_pix = (x,y)     # 记录点击的像素坐标

        # Get click point in camera coordinates
        # 获取点击点的深度值（相机深度图）
        click_z = camera_depth_img[y][x] * robot.cam_depth_scale        # 深度值（乘以深度缩放因子）
        # 利用相机内参（robot.cam_intrinsics）将像素坐标转换为相机坐标系3D坐标
        click_x = np.multiply(x-robot.cam_intrinsics[0][2],click_z/robot.cam_intrinsics[0][0])      # x坐标
        click_y = np.multiply(y-robot.cam_intrinsics[1][2],click_z/robot.cam_intrinsics[1][1])      # y坐标
        if click_z == 0:
            return
        click_point = np.asarray([click_x,click_y,click_z])     # 相机坐标系下的3D点
        click_point.shape = (3,1)

        # Convert camera to robot coordinates
        # camera2robot = np.linalg.inv(robot.cam_pose)
        camera2robot = robot.cam_pose       # 相机到机器人的位姿矩阵
        # 坐标转换：机器人坐标系 = 相机位姿旋转矩阵 × 相机坐标 + 相机位姿平移向量
        target_position = np.dot(camera2robot[0:3,0:3],click_point) + camera2robot[0:3,3:]
        # 提取机器人坐标系下的(x,y,z)
        target_position = target_position[0:3,0]
        print(target_position)
        print(target_position.shape)
        destination=np.append(target_position,tool_orientation)
        robot.plane_grasp([target_position[0],target_position[1],target_position[2]])       # 调用抓取函数


# Show color and depth frames
cv2.namedWindow('color')
cv2.setMouseCallback('color', mouseclick_callback)
cv2.namedWindow('depth')

while True:
    camera_color_img, camera_depth_img = robot.get_camera_data()
    bgr_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
    if len(click_point_pix) != 0:
        bgr_data = cv2.circle(bgr_data, click_point_pix, 7, (0,0,255), 2)
    cv2.imshow('color', bgr_data)
    cv2.imshow('depth', camera_depth_img)
    
    if cv2.waitKey(1) == ord('c'):
        break

cv2.destroyAllWindows()
