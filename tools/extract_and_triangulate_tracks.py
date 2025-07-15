import argparse
import os
import sys
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

# --- 导入所需库 ---
from kornia.feature import LightGlue, DISK
from kornia.geometry import triangulate_points
from kornia.io import ImageLoadType, load_image

# --- 将项目根目录添加到Python路径中，以便导入FSGS的模块 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scene.dataset_readers import readColmapSceneInfo


def build_tracks(pairwise_matches, num_images):
    """从成对的匹配中构建全局特征轨迹"""
    parent = {}

    def find(i):
        if i not in parent: parent[i] = i
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: parent[root_j] = root_i

    for (img1_idx, img2_idx), matches in pairwise_matches.items():
        for p1_idx, p2_idx in matches:
            union((img1_idx, p1_idx), (img2_idx, p2_idx))

    tracks = defaultdict(list)
    for node in parent:
        track_id = find(node)
        tracks[track_id].append(node)

    return list(tracks.values())


def main(args):
    # --- 1. 初始化模型 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    disk = DISK.from_pretrained("depth").to(device).eval()
    lightglue = LightGlue(features='disk', depth_confidence=-1, width_confidence=-1).to(device).eval()

    # --- 2. 加载场景数据 ---
    print("加载相机和图像数据...")
    scene_info = readColmapSceneInfo(args.scene_path)

    sorted_cams = sorted(scene_info.train_cameras, key=lambda x: x.id)
    intrinsics = {cam.id: torch.from_numpy(cam.K).float() for cam in sorted_cams}
    extrinsics = {cam.id: torch.from_numpy(cam.w2c).float() for cam in sorted_cams}
    image_paths = {cam.id: os.path.join(args.scene_path, "images", cam.image_name) for cam in sorted_cams}

    num_images = len(sorted_cams)
    print(f"共找到 {num_images} 张图像。")

    # --- 3. 提取所有图像的特征 ---
    print("提取所有图像的特征...")
    all_features = {}
    for cam in tqdm(sorted_cams, desc="特征提取"):
        img_tensor = load_image(image_paths[cam.id], ImageLoadType.RGB_U8, device=device).float() / 255.0
        with torch.no_grad():
            # 提取特征，包括关键点和描述子
            feats = disk(img_tensor, n=args.num_keypoints, window_size=5, score_threshold=0.001, pad_if_needed=True)
        all_features[cam.id] = feats

    # --- 4. 逐对图像进行特征匹配 (修正后的核心逻辑) ---
    print("进行成对特征匹配...")
    pairwise_matches = {}
    cam_ids = [cam.id for cam in sorted_cams]

    for i in tqdm(range(num_images), desc="特征匹配"):
        for j in range(i + 1, num_images):
            id1, id2 = cam_ids[i], cam_ids[j]
            feats1, feats2 = all_features[id1], all_features[id2]

            with torch.no_grad():
                # **修正**: 将完整的特征字典传递给LightGlue
                matches_info = lightglue(feats1, feats2)

            # 获取匹配点的索引
            matches = matches_info['matches'].cpu().numpy()
            if matches.shape[0] > 0:
                pairwise_matches[(id1, id2)] = matches

    # --- 5. 构建全局特征轨迹 ---
    print("构建全局特征轨迹...")
    raw_tracks = build_tracks(pairwise_matches, num_images)
    print(f"初步构建了 {len(raw_tracks)} 条轨迹。")

    # --- 6. 三角化轨迹生成三维锚点并过滤 ---
    print("三角化轨迹并根据重投影误差进行过滤...")
    final_tracks_2d = []
    final_anchors_3d = []

    for track in tqdm(raw_tracks, desc="三角化与过滤"):
        if len(track) < args.min_track_length:
            continue

        points_2d_track = []
        view_indices_track = []
        proj_matrices_track = []

        for img_idx, point_idx in track:
            points_2d_track.append(all_features[img_idx]['keypoints'][point_idx])
            view_indices_track.append(img_idx)

            K = intrinsics[img_idx].to(device)
            W2C = extrinsics[img_idx].to(device)
            P = K @ W2C
            proj_matrices_track.append(P)

        points_2d_tensor = torch.stack(points_2d_track).to(device)
        proj_matrices_tensor = torch.stack(proj_matrices_track)

        try:
            points_3d_homo = triangulate_points(proj_matrices_tensor, points_2d_tensor.T)
            point_3d = (points_3d_homo[:3] / points_3d_homo[3]).T.squeeze(0)
        except Exception:
            continue

        reprojected_points = (proj_matrices_tensor @ torch.cat([point_3d, torch.ones(1).to(device)])).T
        reprojected_points = reprojected_points[:, :2] / reprojected_points[:, 2:]
        reprojection_error = torch.linalg.norm(points_2d_tensor - reprojected_points, dim=1).mean()

        if reprojection_error.item() < args.reproj_error_threshold:
            final_anchors_3d.append(point_3d.cpu().numpy())
            track_info = np.zeros((len(view_indices_track), 3))
            track_info[:, 0] = view_indices_track
            track_info[:, 1:] = points_2d_tensor.cpu().numpy()
            final_tracks_2d.append(track_info)

    print(f"过滤后剩余 {len(final_anchors_3d)} 条有效轨迹。")

    # --- 7. 保存到文件 ---
    if not final_anchors_3d:
        print("警告: 没有找到有效的轨迹，不生成轨迹文件。")
        return

    output_path = Path(args.scene_path) / "tracks.h5"
    print(f"保存轨迹数据到 {output_path}...")

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('anchors_3d', data=np.array(final_anchors_3d))
        dt = h5py.special_dtype(vlen=np.dtype('float32'))
        dset = f.create_dataset('tracks_2d', (len(final_tracks_2d),), dtype=dt)
        for i, track_info in enumerate(final_tracks_2d):
            dset[i] = track_info.flatten()

    print("第一步完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为GeoTrack-GS预处理特征轨迹")
    parser.add_argument("-s", "--scene_path", type=str, required=True,
                        help="FSGS场景的根目录 (包含colmap/sparse/0文件夹)")
    parser.add_argument("--num_keypoints", type=int, default=2048, help="每张图像提取的特征点数量")
    parser.add_argument("--min_track_length", type=int, default=2, help="一条有效轨迹所需的最少视图数量")
    parser.add_argument("--reproj_error_threshold", type=float, default=2.0,
                        help="用于过滤轨迹的最大平均重投影误差(像素)")
    args = parser.parse_args()
    main(args)
