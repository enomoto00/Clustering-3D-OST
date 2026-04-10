# models/ffnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ocean_model(nn.Module):
    """
    Per-pixel FFNN via 1x1 convs:
    (B, C_in=5, H, W) -> 256 -> 128 -> 64 -> 32 -> (B, C_out=73, H, W)
    """
    def __init__(self, in_channels: int = 5, out_channels: int = 73, dropout_p: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        p = dropout_p

        # 1x1 conv 等价于像元上的线性层；空间维度不变
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(128,  64, kernel_size=1, bias=True)
        self.conv4 = nn.Conv2d( 64,  32, kernel_size=1, bias=True)
        self.out   = nn.Conv2d( 32, out_channels, kernel_size=1, bias=True)

        # 对通道做 dropout（空间一致），比普通 Dropout 更适合栅格图
        self.drop = nn.Dropout2d(p)

        # Kaiming 初始化（适配 ReLU）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容 (B, C) / (B, C, 1, 1) / (B, C, H, W)
        if x.dim() == 2:              # (B, C)
            x = x.unsqueeze(-1).unsqueeze(-1)  # -> (B, C, 1, 1)
        elif x.dim() == 3:            # (B, C, L) 视作 (H=1, W=L)
            x = x.unsqueeze(2)        # -> (B, C, 1, L)
        elif x.dim() == 4:
            pass
        else:
            raise RuntimeError(f"Unexpected input shape: {tuple(x.shape)}")

        if x.size(1) != self.in_channels:
            raise RuntimeError(f"Expect input channels={self.in_channels}, got {x.size(1)}")

        x = F.relu(self.conv1(x)); x = self.drop(x)
        x = F.relu(self.conv2(x)); x = self.drop(x)
        x = F.relu(self.conv3(x)); x = self.drop(x)
        x = F.relu(self.conv4(x)); x = self.drop(x)
        x = self.out(x)  # (B, out_channels, H, W)
        return x

