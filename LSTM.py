import torch
import torch.nn as nn
from typing import Optional


class OceanLSTM(nn.Module):
    """
    输入:  x 形状 (B, in_channels, H, W)     —— 例如 5 个输入通道
    输出:  y 形状 (B, out_channels, H, W)    —— 例如 73 个深度层

    思路:
    - 先用 1x1 Conv 将 5 通道压成一个 latent 向量(每个像素一个向量)。
    - 用这个 latent 初始化 LSTM 的 h0/c0。
    - LSTM 解码长度 = out_channels，每一步输出一个标量(该深度层的预测)。
    - 使用可学习的“深度位置嵌入”作为 LSTM 的逐步输入，确保不同深度有可区分的位置特征。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 将输入的 5 通道映射到隐藏维度（逐像素）
        self.encoder = nn.Conv2d(in_channels, hidden_size, kernel_size=1, bias=True)

        # 用于将编码向量投到 LSTM 初始状态 h0 / c0
        self.h0_proj = nn.Linear(hidden_size, num_layers * hidden_size)
        self.c0_proj = nn.Linear(hidden_size, num_layers * hidden_size)

        # 深度位置嵌入，作为 LSTM 每一步的输入 (seq_len=out_channels)
        self.depth_embed = nn.Embedding(out_channels, hidden_size)

        # LSTM 解码器（按深度序列展开）
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p if num_layers > 1 else 0.0,
        )

        # 将 LSTM 的每步隐状态映射成标量输出
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        return: (B, out_channels, H, W)
        """
        B, _, H, W = x.shape

        # (B, hidden, H, W)
        enc = self.encoder(x)

        # 展平空间维度: (B*H*W, hidden)
        enc_flat = enc.permute(0, 2, 3, 1).reshape(B * H * W, self.hidden_size)

        # 初始状态 h0/c0: (num_layers, B*H*W, hidden)
        h0 = self.h0_proj(enc_flat).view(B * H * W, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        c0 = self.c0_proj(enc_flat).view(B * H * W, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()

        # 构造解码输入: 用 depth embedding 作为每步输入
        # depth tokens: (out_channels, hidden)
        depth_tokens = self.depth_embed(torch.arange(self.out_channels, device=x.device))  # (D, hidden)
        # 扩展到 batch*space: (D, B*H*W, hidden)
        dec_in = depth_tokens[:, None, :].expand(self.out_channels, B * H * W, self.hidden_size)

        # 运行 LSTM 解码
        dec_out, _ = self.decoder(dec_in, (h0, c0))  # dec_out: (D, B*H*W, hidden)

        # 映射到标量并还原形状
        y_flat = self.head(dec_out).squeeze(-1)               # (D, B*H*W)
        y = y_flat.permute(1, 0).contiguous().view(B, H, W, self.out_channels)  # (B, H, W, D)
        y = y.permute(0, 3, 1, 2).contiguous()                # (B, D, H, W)

        return y


def ocean_model(in_channels: int, out_channels: int, dropout_p: float = 0.2) -> nn.Module:
    """
    便捷构造函数：可按需改 hidden_size / num_layers
    """
    return OceanLSTM(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=64,
        num_layers=2,
        dropout_p=dropout_p,
    )
