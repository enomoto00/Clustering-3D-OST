# # model.py  —— 深度当作输出通道: [B, D, H, W] (= [B, 73, 21, 21])

# from typing import Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ---------- 基础模块 ----------
# def _lecun_init(m: nn.Module):
#     if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
#         nn.init.kaiming_normal_(m.weight, nonlinearity="linear", mode="fan_in")
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     return m


# class ConvBlock1D(nn.Module):
#     def __init__(self, in_ch, out_ch, use_bn: bool = True, dropout_p: float = 0.3):
#         super().__init__()
#         self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
#         self.bn = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
#         self.act = nn.GELU()
#         self.drop = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()
#         _lecun_init(self.conv)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.drop(x)
#         return x


# class DeconvBlock1D(nn.Module):
#     def __init__(self, in_ch, out_ch, stride=2, use_bn=True, dropout_p=0.3):
#         super().__init__()
#         self.tconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=3, stride=stride,
#                                         padding=1, output_padding=stride - 1)
#         _lecun_init(self.tconv)
#         self.refine = ConvBlock1D(out_ch, out_ch, use_bn=use_bn, dropout_p=dropout_p)

#     def forward(self, x):
#         x = self.tconv(x)
#         x = self.refine(x)
#         return x


# # ---------- 核心：按像素预测纵剖面 ----------
# class OCNN1D(nn.Module):
#     """
#     输入:  [B, N]
#     输出:  [B, M, D]  (默认 D=73)
#     """
#     def __init__(self,
#                  in_features: int,
#                  out_vars: int,
#                  target_depth_len: int = 73,
#                  base_channels: int = 128,
#                  seed_len: int = 19,
#                  use_bn: bool = True,
#                  dropout_p: float = 0.3,
#                  use_maxpool: bool = False):
#         super().__init__()
#         self.target_depth_len = target_depth_len
#         self.seed_len = seed_len

#         self.fc_seed = nn.Linear(in_features, base_channels * seed_len)
#         _lecun_init(self.fc_seed)

#         ch = base_channels
#         blocks = []
#         for _ in range(3):
#             blocks.append(DeconvBlock1D(ch, ch, stride=2, use_bn=use_bn, dropout_p=dropout_p))
#             if use_maxpool:
#                 blocks.append(nn.MaxPool1d(kernel_size=3, stride=3))
#         self.up_path = nn.Sequential(*blocks)

#         self.tail = nn.Sequential(
#             ConvBlock1D(ch, ch, use_bn=use_bn, dropout_p=dropout_p),
#             nn.Conv1d(ch, out_vars, kernel_size=1, stride=1),
#         )
#         _lecun_init(self.tail[-1])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, N]
#         B = x.size(0)
#         x = self.fc_seed(x).view(B, -1, self.seed_len)   # [B, C, L0]
#         x = self.up_path(x)                              # [B, C, L~]
#         L = x.size(-1)
#         if L < self.target_depth_len:
#             x = F.interpolate(x, size=self.target_depth_len, mode="linear", align_corners=False)
#         else:
#             x = x[..., : self.target_depth_len]
#         x = self.tail(x)                                 # [B, out_vars, D]
#         return x


# class OCNNPerPixelDepthAsChannel(nn.Module):
#     """
#     输入:  [B, C_in, H, W]，其中 H=W=21
#     输出:  [B, D, H, W]，其中 D=depth_len=73
#     逻辑:  每个像素共享同一个 OCNN1D（out_vars=1），输出一条长度为 D 的剖面，
#           然后把 D 维放到通道维，得到 [B, D, H, W]
#     """
#     def __init__(self,
#                  in_channels: int,
#                  depth_len: int = 73,
#                  base_channels: int = 128,
#                  seed_len: int = 19,
#                  use_bn: bool = True,
#                  dropout_p: float = 0.3):
#         super().__init__()
#         self.depth_len = depth_len
#         # 注意：这里 out_vars=1（每个像素只预测一个变量的 D 层剖面）
#         self.core = OCNN1D(
#             in_features=in_channels,
#             out_vars=1,
#             target_depth_len=depth_len,
#             base_channels=base_channels,
#             seed_len=seed_len,
#             use_bn=use_bn,
#             dropout_p=dropout_p,
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, C_in, H, W]
#         B, C, H, W = x.shape
#         x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)  # [B*H*W, C_in]
#         y = self.core(x)                                           # [B*H*W, 1, D]
#         y = y.squeeze(1)                                           # [B*H*W, D]
#         y = y.view(B, H, W, self.depth_len)                        # [B, H, W, D]
#         y = y.permute(0, 3, 1, 2).contiguous()                     # [B, D, H, W]
#         return y


# model.py — per-pixel OCNN: 输入 [B, C_in, H=21, W=21] → 输出 [B, D=output_c, H, W]

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lecun_init(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="linear", mode="fan_in")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    return m


class ConvBNAct(nn.Module):
    """Conv1d(k=3,s=1,p=1) + BN + GELU (+ Dropout 可选)"""
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()
        _lecun_init(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class PixelOCNN1D(nn.Module):
    """
    单像素 1D OCNN（优先 3 次池化；若 3 次后长度 < min_len_after，则退到 2 次）
      输入:  [B, N]
      输出:  [B, 1, D]  (D=output_c)
    """
    def __init__(self, in_features: int, depth_len: int, dropout_p: float = 0.2,
                 min_len_after: int = 3):   # 阈值：3 更稳；也可改成 2/4
        super().__init__()
        if depth_len <= 0:
            raise ValueError("depth_len (output_c) must be a positive integer.")
        self.D = depth_len
        self.min_len_after = max(1, int(min_len_after))

        # 1) FC: [B,N] -> [B,D]
        self.fc_seed = nn.Linear(in_features, self.D)
        _lecun_init(self.fc_seed)

        # 2) DeConv: 1->64 通道，长度保持 D
        self.deconv = nn.ConvTranspose1d(1, 64, kernel_size=3, stride=1, padding=1)
        _lecun_init(self.deconv)

        # ---- 计划池化次数：优先 3 次，否则 2 次（ceil_mode=True）----
        def after_pool(L, times):
            for _ in range(times):
                if L < 2:           # 长度不足以再池化
                    return 0
                L = (L + 1) // 2    # ceil(L/2)
            return L

        L3 = after_pool(self.D, 3)   # 3 次后的长度
        use_3_pools = (L3 >= self.min_len_after)
        pools_used = 3 if use_3_pools else 2

        # 记录每一层是否池化（最多三层）
        self.pool_flags = [False, False, False]
        for i in range(pools_used):
            self.pool_flags[i] = True

        # 计算最终长度 L_after（用于决定 fc_out 维度）
        L = self.D
        for i in range(3):
            if self.pool_flags[i]:
                L = (L + 1) // 2
        self.L_after = L                       # 例如 D=13 -> 两次池化后 4
        self.last_ch = 8

        # 3) 三个 Conv + (可选 Pool)
        self.conv1 = ConvBNAct(64, 32, dropout_p)
        self.pool1 = nn.MaxPool1d(2, 2, ceil_mode=True) if self.pool_flags[0] else nn.Identity()

        self.conv2 = ConvBNAct(32, 16, dropout_p)
        self.pool2 = nn.MaxPool1d(2, 2, ceil_mode=True) if self.pool_flags[1] else nn.Identity()

        self.conv3 = ConvBNAct(16,  8, dropout_p)
        self.pool3 = nn.MaxPool1d(2, 2, ceil_mode=True) if self.pool_flags[2] else nn.Identity()

        # 4) FC: last_ch * L_after -> D
        self.fc_out = nn.Linear(self.last_ch * self.L_after, self.D)
        _lecun_init(self.fc_out)

        # 方便你在日志里确认本次实例用了几次池化
        self.pools_used = pools_used

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N]
        B = x.size(0)
        x = self.fc_seed(x)              # [B, D]
        x = x.unsqueeze(1)               # [B, 1, D]
        x = F.gelu(self.deconv(x))       # [B, 64, D]

        x = self.conv1(x); x = self.pool1(x)
        x = self.conv2(x); x = self.pool2(x)
        x = self.conv3(x); x = self.pool3(x)

        x = x.flatten(1)                 # [B, 8*L_after]
        x = self.fc_out(x)               # [B, D]
        return x.unsqueeze(1)            # [B, 1, D]



class OCNNPerPixelDepthAsChannel(nn.Module):
    """
    网格封装：对 H×W 每个像素调用同一个 PixelOCNN1D
      输入:  [B, C_in, H, W]   (C_in=输入变量数，H=W=21)
      输出:  [B, D, H, W]      (D=output_c)
    """
    def __init__(self, in_channels: int = 5, depth_len: int = 73, H: int = 21, W: int = 21, dropout_p: float = 0.2):
        super().__init__()
        self.D, self.H, self.W = depth_len, H, W
        self.core = PixelOCNN1D(in_features=in_channels, depth_len=depth_len, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        B, C, H, W = x.shape
        assert (H, W) == (self.H, self.W), f"expect (H,W)=({self.H},{self.W}), got ({H},{W})"
        # [B,C,H,W] -> [B*H*W, C]
        x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        # per-pixel 1D OCNN -> [B*H*W, 1, D]
        y = self.core(x)
        # 还原到网格: [B, D, H, W]
        y = y.view(B, H, W, 1, self.D).permute(0, 3, 4, 1, 2).contiguous().squeeze(1)
        return y



