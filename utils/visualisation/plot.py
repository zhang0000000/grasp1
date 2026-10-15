import warnings
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_processing.grasp import detect_grasps
from utils.dataset_processing.grasp import GraspTriangle

warnings.filterwarnings("ignore")


def plot_results(
        fig,
        rgb_img,
        grasp_q_img,
        grasp_angle_img,
        depth_img=None,
        no_grasps=1,
        grasp_width_img=None
):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')

    if depth_img is not None:
        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(depth_img, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 4)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 5)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 6)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)

    plt.pause(0.1)
    fig.canvas.draw()


def plot_grasp(
        fig,
        grasps=None,
        save=False,
        rgb_img=None,
        grasp_q_img=None,
        grasp_angle_img=None,
        no_grasps=1,
        grasp_width_img=None
):
    """绘制三指抓取的三角形框"""
    if grasps is None:
        grasps = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()   # 启动matplotlib的交互模式，画动态图
    plt.clf()   # 清除当前图形的内容，但不关闭图形窗口，允许该窗口被重复使用

    ax = plt.subplot(111)   #创建一个新的图形窗口及一个或多个子图
    ax.imshow(rgb_img)

    # 只处理前10个抓取姿态
    top_grasps = grasps[:10]
    num_grasps = len(top_grasps)

    # 生成从绿到红的颜色渐变（使用RdYlGn colormap的反转）
    # 最佳姿态（第0个）为绿色，最差（第9个）为红色
    colors = cm.RdYlGn(np.linspace(1, 0, num_grasps))

    for idx, grasp in enumerate(top_grasps):
        if isinstance(grasp, GraspTriangle):
            vertices = grasp.vertices
            vertices = np.vstack([vertices, vertices[0]])  # 闭合三角形
            # 使用对应索引的颜色绘制
            ax.plot(
                vertices[:, 1],  # x坐标
                vertices[:, 0],  # y坐标
                '-', 
                linewidth=2, 
                color=colors[idx],
                label=f"Grasp {idx+1}" if idx == 0 else ""  # 只给最佳姿态加标签
            )

    # for grasp in grasps:
    #     if isinstance(grasp, GraspTriangle):
    #         # 获取三角形顶点并闭合（最后一个点连回第一个点）
    #         vertices = grasp.vertices
    #         vertices = np.vstack([vertices, vertices[0]])  # 闭合三角形
    #         # 绘制蓝色三角形（注意：图像中x是列，y是行，所以用vertices[:,1]（x）和vertices[:,0]（y））
    #         ax.plot(vertices[:, 1], vertices[:, 0], 'b-', linewidth=2)

    # 给最佳姿态添加图例
    if num_grasps > 0:
        ax.legend(['Best Grasp'], loc='upper right')

    plt.pause(2)
    fig.canvas.draw()

    if save:
        time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        fig.savefig('results/{}.png'.format(time))  # 保存可视化结果


def plot_grasp_2(
# def plot_grasp(
        fig,
        grasps=None,
        save=False,
        rgb_img=None,
        grasp_q_img=None,
        grasp_angle_img=None,
        no_grasps=1,
        grasp_width_img=None
):
    """
    Plot the output grasp of a network
    :param fig: Figure to plot the output
    :param grasps: grasp pose(s)
    :param save: Bool for saving the plot
    :param rgb_img: RGB Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    if grasps is None:
        grasps = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()

    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in grasps:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')

    plt.pause(2)
    fig.canvas.draw()

    if save:
        time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        fig.savefig('results/{}.png'.format(time))


def save_results(rgb_img, grasp_q_img, grasp_angle_img, depth_img=None, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')
    fig.savefig('results/rgb.png')

    if depth_img.any():
        fig = plt.figure(figsize=(10, 10))
        plt.ion()
        plt.clf()
        ax = plt.subplot(111)
        ax.imshow(depth_img, cmap='gray')
        for g in gs:
            g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')
        fig.savefig('results/depth.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')
    fig.savefig('results/grasp.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('results/quality.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('results/angle.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('results/width.png')

    fig.canvas.draw()
    plt.close(fig)
