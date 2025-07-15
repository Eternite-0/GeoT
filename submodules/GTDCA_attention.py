# submodules/GTDCA_attention.py (修正版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =================================================================================
# 辅助函数 & 可变形采样核心逻辑
# =================================================================================

def _get_activation_fn(activation):
    """根据字符串返回一个激活函数"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"激活函数应为 relu/gelu, 而不是 {activation}.")


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    多尺度可变形注意力的核心PyTorch实现。
    """
    N, S, M, C = value.shape
    _, Lq, M, L, P, _ = sampling_locations.shape

    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(N * M, C, H_, W_)

        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        # ===================================================================
        # --- 错误修正之处 ---
        # 原来的代码: sampling_grid_l_.unsqueeze(1) 错误地将grid变成了5维
        # F.grid_sample 需要一个4维的grid: [N, H_out, W_out, 2]
        # sampling_grid_l_ 的形状已经是正确的 [N*M, Lq, P, 2]，无需增加维度。
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        # ===================================================================

        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(N * M, 1, Lq, L * P)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N, M, C, Lq)

    return output.transpose(1, 2).contiguous()


# =================================================================================
# GT-DCA 模块定义 (这部分无需更改)
# =================================================================================

class GTDCALayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=1, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points

        self.guidance_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.guidance_norm = nn.LayerNorm(d_model)
        self.trajectory_projection = nn.Linear(2, d_model)

        self.sampling_offset_prediction = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, value, reference_points, trajectory_points):
        B, N_g, C = query.shape
        _, _, H, W = value.shape
        device = query.device

        trajectory_features = self.trajectory_projection(trajectory_points)
        guidance_context, _ = self.guidance_cross_attn(query, trajectory_features, trajectory_features)
        guided_query = self.guidance_norm(query + guidance_context)

        sampling_offsets = self.sampling_offset_prediction(guided_query).view(B, N_g, self.n_heads, self.n_levels,
                                                                              self.n_points, 2)
        attention_weights = self.attention_weights(guided_query).view(B, N_g, self.n_heads, -1)
        attention_weights = F.softmax(attention_weights, -1).view(B, N_g, self.n_heads, self.n_levels, self.n_points)

        sampling_locations = reference_points.unsqueeze(2).unsqueeze(3).unsqueeze(4) + sampling_offsets

        value_reshaped = value.flatten(2).transpose(1, 2)
        value_reshaped = value_reshaped.view(B, H * W, self.n_heads, C // self.n_heads)

        value_spatial_shapes = torch.as_tensor([[H, W]], dtype=torch.long, device=device)

        sampled_features = ms_deform_attn_core_pytorch(
            value_reshaped, value_spatial_shapes, sampling_locations, attention_weights
        )

        sampled_features = sampled_features.view(B, N_g, C)
        output = self.output_proj(sampled_features)

        return output


class GTDCA(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=1, n_points=4):
        super().__init__()
        self.layer = GTDCALayer(d_model, n_heads, n_levels, n_points)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, value, reference_points, trajectory_points):
        output = self.layer(query, value, reference_points, trajectory_points)
        return self.norm(query + output)

