import argparse
import os
import sys
import h5py
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# --- 将项目根目录添加到Python路径中 ---
# 这使得我们可以直接复用FSGS的数据加载逻辑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from scene.dataset_readers import readColmapSceneInfo


def visualize_3d_anchors(anchors_3d, colmap_points=None):
    """
    使用 Open3D 可视化三维锚点。
    可以额外加载COLMAP生成的稀疏点云作为参考。
    """
    if anchors_3d.shape[0] == 0:
        print("错误：没有找到3D锚点进行可视化。")
        return

    # 创建我们的锚点点云对象
    anchor_pcd = o3d.geometry.PointCloud()
    anchor_pcd.points = o3d.utility.Vector3dVector(anchors_3d)
    # 给我们的锚点上色，例如红色，以便区分
    anchor_pcd.paint_uniform_color([1.0, 0, 0])  # Red

    geometries_to_draw = [anchor_pcd]

    print("\n--- 3D 可视化指南 ---")
    print("红色点: 您通过脚本生成的3D锚点。")

    # 如果提供了COLMAP点云作为参考
    if colmap_points is not None and colmap_points.shape[0] > 0:
        colmap_pcd = o3d.geometry.PointCloud()
        colmap_pcd.points = o3d.utility.Vector3dVector(colmap_points)
        colmap_pcd.paint_uniform_color([0.8, 0.8, 0.8])  # Gray
        geometries_to_draw.append(colmap_pcd)
        print("灰色点: COLMAP生成的原始稀疏点云 (作为参考)。")

    print("\n正在打开Open3D窗口... (按 'q' 键关闭窗口)")
    o3d.visualization.draw_geometries(geometries_to_draw, window_name="3D Anchor Points Visualization")


def visualize_2d_tracks_on_images(tracks_h5_path, scene_path, num_images_to_show=3):
    """在原始图像上可视化二维轨迹点"""
    print("\n--- 2D 可视化指南 ---")
    print("将在几张示例图片上用 绿色十字 标出找到的2D特征点。")

    # 加载轨迹数据
    try:
        with h5py.File(tracks_h5_path, 'r') as f:
            if 'tracks_2d' not in f or len(f['tracks_2d']) == 0:
                print("错误：在H5文件中没有找到2D轨迹数据。")
                return
            # 从HDF5的可变长度数据中恢复轨迹
            tracks_2d_data = [np.array(track).reshape(-1, 3) for track in f['tracks_2d']]
    except Exception as e:
        print(f"读取轨迹文件失败: {e}")
        return

    # 加载图像路径
    scene_info = readColmapSceneInfo(scene_path)
    sorted_cams = sorted(scene_info.train_cameras, key=lambda x: x.id)
    image_paths = {cam.id: os.path.join(scene_path, "images", cam.image_name) for cam in sorted_cams}

    # 为每个视图收集所有特征点
    points_per_view = {i: [] for i in range(len(sorted_cams))}
    for track in tracks_2d_data:
        for view_idx, x, y in track:
            points_per_view[int(view_idx)].append((x, y))

    # 可视化前几张图像
    for i in range(min(num_images_to_show, len(sorted_cams))):
        cam_id = sorted_cams[i].id
        img_path = image_paths[cam_id]

        if not os.path.exists(img_path):
            print(f"警告: 找不到图像 {img_path}")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        points_on_this_image = points_per_view.get(cam_id, [])

        # 在图像上绘制特征点
        for x, y in points_on_this_image:
            cv2.drawMarker(img, (int(x), int(y)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10,
                           thickness=1)

        # 使用 Matplotlib 显示
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.title(f"视图 {i} ({os.path.basename(img_path)}) 上的2D轨迹点 (共 {len(points_on_this_image)} 个)",
                  fontsize=16)
        plt.axis('off')
        plt.show()


def main(args):
    scene_path = Path(args.scene_path)
    h5_path = scene_path / "tracks.h5"

    if not h5_path.exists():
        print(f"错误: 在 '{scene_path}' 目录下找不到 'tracks.h5' 文件。")
        print("请先运行 `extract_and_triangulate_tracks.py` 脚本。")
        return

    # --- 1. 可视化3D锚点 ---
    with h5py.File(h5_path, 'r') as f:
        anchors_3d = f['anchors_3d'][:]

    # 尝试加载COLMAP点云作为参考
    colmap_points_path = scene_path / "sparse" / "0" / "points3D.bin"
    colmap_points = None
    if colmap_points_path.exists():
        try:
            # 复用FSGS的逻辑来读取.bin文件
            from utils.general_utils import read_points3d_binary
            colmap_data = read_points3d_binary(str(colmap_points_path))
            colmap_points = np.array([p.xyz for p in colmap_data.values()])
        except Exception as e:
            print(f"读取COLMAP点云失败: {e}")

    visualize_3d_anchors(anchors_3d, colmap_points)

    # --- 2. 可视化2D轨迹 ---
    visualize_2d_tracks_on_images(str(h5_path), str(scene_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化验证GeoTrack-GS的几何数据")
    parser.add_argument("-s", "--scene_path", type=str, required=True, help="FSGS场景的根目录 (例如 ./data/nerf/fox)")
    args = parser.parse_args()
    main(args)
