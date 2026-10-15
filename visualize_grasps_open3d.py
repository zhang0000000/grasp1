import numpy as np
import open3d as o3d
import torch
from inference.post_process import post_process_output
from utils.dataset_processing.grasp import detect_grasps
from utils.data.camera_data import CameraData
from plane_grasp_real import PlaneGraspClass

def convert_2d_to_3d(row, col, depth_img, intrinsic):
    """将图像坐标（row, col）转换为3D相机坐标"""
    z = depth_img[row, col]  # 深度值（单位：米）
    if z <= 0:
        return None
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    cx, cy = intrinsic[0][2], intrinsic[1][2]
    x = (col - cx) * z / fx
    y = (row - cy) * z / fy
    return np.array([x, y, z])

def create_rgb_depth_visualization(rgb, depth):
    """创建RGBD图像可视化对象"""
    # 转换RGB图像为Open3D格式
    rgb_o3d = o3d.geometry.Image(rgb)
    
    # 转换深度图为Open3D格式（归一化以便可视化）
    depth_normalized = (depth / np.max(depth) * 255).astype(np.uint8)
    depth_o3d = o3d.geometry.Image(depth_normalized)
    
    # 创建RGBD图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=1.0, convert_rgb_to_intensity=False)
    
    return rgbd_image

def visualize_grasps():
    # 初始化抓取检测类
    grasp_class = PlaneGraspClass(
        saved_model_path='trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93',
        visualize=False
    )

    # 获取RGB、深度图和相机内参
    rgb, depth = grasp_class.ur_robot.get_camera_data()
    depth_meters = depth * grasp_class.cam_depth_scale  # 转换为米

    # 关键修复：为深度图增加通道维度（从2D变为3D：(H, W) -> (H, W, 1)）
    depth_meters = np.expand_dims(depth_meters, axis=2)  # 添加这一行

    intrinsic = grasp_class.intrinsic  # 相机内参矩阵

    # 预处理数据并检测前10个抓取姿态
    # 此时传入的depth_meters是3D数组，可正常转置
    x, depth_img, rgb_img = grasp_class.cam_data.get_data(rgb=rgb, depth=depth_meters)

    if depth_img is not None:
        # 仅当数组是2维时才添加通道维度（避免重复添加）
        if len(depth_img.shape) == 2:
            depth_img = np.expand_dims(depth_img, axis=2)
        depth_img = depth_img.transpose((2, 0, 1))  # 此时维度为(1, H, W)，转置有效

    with torch.no_grad():
        xc = x.to(grasp_class.device)
        pred = grasp_class.model.predict(xc)
    q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
    grasps = detect_grasps(q_img, ang_img, width_img, no_grasps=10)  # 确保获取前10个
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Grasp Visualization", width=1280, height=720)
    
    # 添加详细点云（减少降采样步长以提高细节）
    pcd = o3d.geometry.PointCloud()
    points = []
    colors = []
    # 降低采样步长以获取更详细的点云
    for i in range(0, depth.shape[0], 2):  # 步长从10改为2
        for j in range(0, depth.shape[1], 2):
            pt = convert_2d_to_3d(i, j, depth_meters, intrinsic)
            if pt is not None:
                points.append(pt)
                # 添加RGB颜色
                colors.append(rgb[i, j] / 255.0)  # 归一化到0-1范围
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)
    
    # 逐个添加抓取姿态（优化显示效果）
    for idx, grasp in enumerate(grasps):
        center_2d = grasp.center  # (row, col)
        center_3d = convert_2d_to_3d(center_2d[0], center_2d[1], depth_meters, intrinsic)
        if center_3d is None:
            continue
        
        # 调整抓取框大小（放大半径以便更好地可视化）
        radius = grasp.radius * 1.5  # 适当放大半径
        angle = grasp.angle
        
        # 生成三指抓取框顶点
        angles = [angle + 2 * np.pi / 3 * i for i in range(3)]
        vertices = []
        for a in angles:
            dx = radius * np.cos(a)
            dy = radius * np.sin(a)
            vertices.append(center_3d + np.array([dx, dy, 0]))
        
        # 创建三角形网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
        
        # 为不同抓取姿态设置渐变颜色（从绿到红）
        color_ratio = idx / len(grasps)
        mesh.paint_uniform_color([color_ratio, 1 - color_ratio, 0])  # 绿色到红色渐变
        
        # 添加线框以增强可见性
        lines = [[0, 1], [1, 2], [2, 0]]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vertices),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.paint_uniform_color([0, 0, 0])  # 黑色线框
        line_set.line_width = 3  # 线宽
        
        vis.add_geometry(mesh)
        vis.add_geometry(line_set)
    
    # 创建并显示RGBD图像（在新窗口）
    rgbd_image = create_rgb_depth_visualization(rgb, depth)
    o3d.visualization.draw_geometries([
        o3d.geometry.Image(rgbd_image.color),
        o3d.geometry.Image(rgbd_image.depth)
    ], window_name="RGBD Image")
    
    # 显示点云和抓取框
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualize_grasps()