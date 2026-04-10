import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBNLayer,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class CombinedConBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CombinedConBNLayer, self).__init__()
        self.layer1 = ConvBNLayer(in_channels, out_channels)
        self.layer2 = ConvBNLayer(out_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels//reduction, bias=False)
        self.fc2 = nn.Linear(in_channels//reduction, in_channels, bias=False)

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        max_pool = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
        channel_att = F.relu(self.fc1(avg_pool)) + F.relu(self.fc1(max_pool))
        channel_att = torch.sigmoid(self.fc2(channel_att)).view(x.size(0), x.size(1), 1, 1)

        return x*channel_att

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = self.conv(torch.cat([avg_pool, max_pool], dim=1))
        return x * spatial_att

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# class ocean_model(nn.Module):
#     def __init__(self, outchannels=1):
#         super(ocean_model, self).__init__()
#         self.ConBN1 = nn.Sequential(
#             ConvBNLayer(in_channels=5, out_channels=32),
#             ConvBNLayer(in_channels=32, out_channels=64)
#         )
#         self.ConBN2 = nn.Sequential(
#             ConvBNLayer(in_channels=64, out_channels=32),
#             ConvBNLayer(in_channels=32, out_channels=64)
#         )
#         self.CombinedConBN = CombinedConBNLayer(in_channels=128, out_channels=64)
#         self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.CBAM1 = CBAM(in_channels=64)
#         self.CBAM2 = CBAM(in_channels=64)
#         self.upsample = nn.Upsample(size=(16, 16), mode='bilinear', align_corners=True)
#         self.Conv = nn.Conv2d(in_channels=64, out_channels=outchannels, kernel_size=1)

#     def forward(self, x):
#         x1 = self.ConBN1(x)
#         x2 = self.CBAM1(x1)
#         x3 = self.MaxPool(x1)
#         x4 = self.ConBN2(x3)
#         x5 = self.CBAM2(x4)
#         x6 = self.upsample(x5)
#         x7 = self.CombinedConBN(x2, x6)
#         x8 = self.Conv(x7)
#         return x8


class ocean_model(nn.Module):
    def __init__(self, outchannels=1):
        super(ocean_model, self).__init__()
        self.ConBN1 = nn.Sequential(
            ConvBNLayer(in_channels=5, out_channels=32),
            ConvBNLayer(in_channels=32, out_channels=64)
        )
        self.ConBN2 = nn.Sequential(
            ConvBNLayer(in_channels=64, out_channels=32),
            ConvBNLayer(in_channels=32, out_channels=64)
        )
        self.CombinedConBN = CombinedConBNLayer(in_channels=128, out_channels=64)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 👈 支持奇数输入
        self.CBAM1 = CBAM(in_channels=64)
        self.CBAM2 = CBAM(in_channels=64)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 👈 动态上采样
        self.Conv = nn.Conv2d(in_channels=64, out_channels=outchannels, kernel_size=1)

    def forward(self, x):
        x1 = self.ConBN1(x)  # [B, 64, 21, 21]
        x2 = self.CBAM1(x1)  # [B, 64, 21, 21]
        x3 = self.MaxPool(x1)  # [B, 64, 11, 11]
        x4 = self.ConBN2(x3)   # [B, 64, 11, 11]
        x5 = self.CBAM2(x4)    # [B, 64, 11, 11]
        x6 = self.upsample(x5)  # [B, 64, 22, 22]

        # 👇 尺寸对齐
        if x6.size()[2:] != x2.size()[2:]:
            diffY = x2.size()[2] - x6.size()[2]
            diffX = x2.size()[3] - x6.size()[3]
            x6 = F.pad(x6, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])  # 反向填充

        x7 = self.CombinedConBN(x2, x6)
        x8 = self.Conv(x7)
        return x8  # [B, 73, 21, 21]


