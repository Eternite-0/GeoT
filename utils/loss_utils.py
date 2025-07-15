#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import kornia.filters as kf


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def huber_loss(error: torch.Tensor, delta: float = 1.0):
    """
    鲁棒的Huber损失实现。
    当误差小于delta时，表现为L2损失；大于delta时，表现为L1损失。
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    return 0.5 * quadratic ** 2 + delta * linear


# --- GeoTrack-GS 几何约束模块 ---

def reprojection_loss(
        gaussians,
        scene_cameras,
        tracks_2d_info: list,
        anchors_3d: torch.Tensor
):
    """
    计算全局重投影一致性损失 L_reproj。
    这是GeoTrack-GS的核心强约束。
    """
    if anchors_3d.shape[0] == 0:
        return torch.tensor(0.0, device="cuda")

    # 1. 获取与3D锚点关联的高斯基元中心μ
    associated_centers = gaussians.get_closest_gaussians_for_anchors(anchors_3d)

    # 2. 准备重投影计算所需的数据
    points_3d_to_project = []
    points_2d_ground_truth = []
    projection_matrices = []

    for i, track_info in enumerate(tracks_2d_info):
        # track_info 的每一行是 (view_idx, x, y)
        num_points_in_track = track_info.shape[0]

        # 将该轨迹对应的单个3D锚点重复N次，N为轨迹长度
        points_3d_to_project.append(associated_centers[i].unsqueeze(0).expand(num_points_in_track, -1))

        # 收集轨迹中的2D真值点
        points_2d_ground_truth.append(torch.from_numpy(track_info[:, 1:]).to("cuda"))

        # 收集对应的相机投影矩阵
        view_indices = track_info[:, 0].astype(int)
        for view_idx in view_indices:
            cam = scene_cameras[view_idx]
            # P = K @ [R|t]
            proj_matrix = cam.full_proj_transform.T
            projection_matrices.append(proj_matrix)

    points_3d_to_project = torch.cat(points_3d_to_project, dim=0)
    points_2d_ground_truth = torch.cat(points_2d_ground_truth, dim=0)
    projection_matrices = torch.stack(projection_matrices, dim=0)

    # 3. 执行重投影
    # 将3D点转换为齐次坐标 [X, Y, Z, 1]
    points_3d_homo = F.pad(points_3d_to_project, (0, 1), mode='constant', value=1.0)

    # P @ p_3d_homo -> [x', y', z']
    projected_points_homo = (projection_matrices @ points_3d_homo.T).T

    # 透视除法 [x'/z', y'/z']
    projected_points_2d = projected_points_homo[:, :2] / projected_points_homo[:, 2:]

    # 4. 计算误差并应用Huber损失
    error = projected_points_2d - points_2d_ground_truth
    loss = huber_loss(error).mean()

    return loss


def compute_feature_reliability_mask(
        tracks_2d_info: list,
        num_views: int,
        height: int,
        width: int,
        device: str = "cuda"
):
    """
    计算特征可靠性掩码 M。
    在有特征点的区域，可靠性高；在无纹理区域，可靠性低。
    """
    reliability_map = torch.zeros((num_views, 1, height, width), device=device)

    for track in tracks_2d_info:
        for view_idx, x, y in track:
            # 将坐标限制在图像范围内
            u, v = int(round(x)), int(round(y))
            if 0 <= u < width and 0 <= v < height:
                reliability_map[int(view_idx), 0, v, u] = 1.0

    # 使用高斯模糊使稀疏的特征点扩散成一个平滑的可靠性区域
    # kernel_size 和 sigma 是超参数，可以根据效果调整
    blurred_map = kf.gaussian_blur2d(reliability_map, kernel_size=(31, 31), sigma=(5.0, 5.0))

    # 归一化到 [0, 1]
    # M 的值越大，代表该区域特征越可靠
    M = (blurred_map - blurred_map.min()) / (blurred_map.max() - blurred_map.min() + 1e-8)

    return M


def hybrid_geometric_loss(
        reproj_loss_val: torch.Tensor,
        original_depth_loss_map: torch.Tensor,
        reliability_mask_M: torch.Tensor
):
    """
    计算创新的混合几何损失。

    Args:
        reproj_loss_val: L_reproj 的标量值 (强约束)
        original_depth_loss_map: FSGS原有的、逐像素的深度损失图谱 (弱约束)
        reliability_mask_M: 特征可靠性掩码
    """
    # 在特征不可靠的区域 (1 - M) 应用深度损失
    # M越大，(1-M)越小，深度损失的权重就越低
    masked_depth_loss = torch.mean((1 - reliability_mask_M) * original_depth_loss_map)

    # 总的几何损失是强约束和弱化的弱约束之和
    total_geom_loss = reproj_loss_val + masked_depth_loss

    return total_geom_loss




