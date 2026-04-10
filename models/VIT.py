"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor

    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # 随机drop一个完整的block，
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, image_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        """
        Map input tensor to patch.
        Args:
            image_size: input image size
            patch_size: patch size
            in_c: number of input channels
            embed_dim: embedding dimension. dimension = patch_size * patch_size * in_c
            norm_layer: The function of normalization
        """
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # The input tensor is divided into patches using 16x16 convolution
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5      # 根号d，缩放因子
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    def __init__(self,
                 dim,                # 输入特征的维度
                 num_heads,          # 多头自注意力机制的头数
                 mlp_ratio=4.,       # MLP 隐藏层大小与输入大小的比率
                 qkv_bias=False,     # 是否在 Q、K、V 计算中使用偏置
                 qk_scale=None,      # Q 和 K 的缩放因子
                 drop_ratio=0.,      # Dropout 比率
                 attn_drop_ratio=0., # 自注意力中的 Dropout 比率
                 drop_path_ratio=0., # 随机深度的 DropPath 比率
                 act_layer=nn.GELU,  # 激活函数
                 norm_layer=nn.LayerNorm):  # 归一化层
        super(Block, self).__init__()
        
        # 第一层归一化
        self.norm1 = norm_layer(dim)
        
        # 自注意力层，包含 Q、K、V 的计算及其后续处理
        self.attn = Attention(dim, 
                              num_heads=num_heads, 
                              qkv_bias=qkv_bias, 
                              qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, 
                              proj_drop_ratio=drop_ratio)
        
        # 随机深度的 DropPath，若 drop_path_ratio 大于 0，则使用 DropPath，否则使用 nn.Identity（不进行任何操作）
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        
        # 第二层归一化
        self.norm2 = norm_layer(dim)
        
        # 计算 MLP 隐藏层的维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # MLP 层，包含激活函数和 Dropout
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # 前向传播过程
        # 1. 先进行自注意力计算，再进行归一化和随机深度处理
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # 2. 然后进行 MLP 计算，归一化和随机深度处理
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
    
class Decoder(nn.Module):
    def __init__(self, embed_dim=36, output_c=1):
        super(Decoder, self).__init__()
        
        # 将特征维度从 embed_dim (36) 转换到 512，然后逐步上采样到 21x21
        self.fc = nn.Linear(embed_dim, 512 * 3 * 3)  # 通过全连接层扩展到更大的特征图

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)  # 3x3 -> 6x6
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0)  # 6x6 -> 12x12
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)   # 12x12 -> 24x24
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0)    # 24x24 -> 48x48

        # 卷积调整到指定输出通道数和大小（21x21）
        self.final_conv = nn.Conv2d(32, out_channels=output_c, kernel_size=3, stride=1, padding=1)  

    def forward(self, x):
        # x 形状为 (batch_size, embed_dim)，例如 (32, 36)
        x = self.fc(x)  # (32, 512 * 3 * 3)
        x = x.view(x.size(0), 512, 3, 3)  # 调整为 (batch_size, 512, 3, 3)

        x = self.deconv1(x)  # (batch_size, 256, 6, 6)
        x = self.deconv2(x)  # (batch_size, 128, 12, 12)
        x = self.deconv3(x)  # (batch_size, 64, 24, 24)
        x = self.deconv4(x)  # (batch_size, 32, 48, 48)

        # 最终卷积调整到指定通道数和更大尺寸 (batch_size, output_c, 48, 48)
        x = self.final_conv(x)  # (batch_size, output_c, 48, 48)

        # 裁剪到 21x21
        x = x[:, :, :21, :21]  # (batch_size, output_c, 21, 21)

        # 如果需要更严格的尺寸对齐，可以使用自适应平均池化
        x = nn.functional.adaptive_avg_pool2d(x, (21, 21))  # (batch_size, output_c, 21, 21)

        return x
    
    
    
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial

class VisionTransformer(nn.Module):  # 定义 VisionTransformer 类，继承自 nn.Module
    def __init__(self, image_size=224, patch_size=16, in_c=5,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0.5, embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None,output_channel=1):
        # num_classes=1000,
        """
        初始化 VisionTransformer 类的参数。

        Args:
            image_size (int, tuple): 输入图像的大小
            patch_size (int, tuple): 每个图像块的大小
            in_c (int): 输入通道数
            num_classes (int): 分类头的类别数??
            embed_dim (int): 嵌入维度
            depth (int): Transformer 的深度（层数）
            num_heads (int): 注意力头的数量
            mlp_ratio (int): MLP 隐藏层维度与嵌入维度的比例
            qkv_bias (bool): 如果为 True，则为 QKV 启用偏置
            qk_scale (float): 如果设置，则覆盖默认的 qk 缩放
            representation_size (Optional[int]): 如果设置，启用并将表示层设置为此值
            distilled (bool): 模型包含蒸馏标记和头部
            drop_ratio (float): dropout 比例
            attn_drop_ratio (float): 注意力 dropout 比例
            drop_path_ratio (float): 随机深度率
            embed_layer (nn.Module): 图像块嵌入层
            norm_layer (nn.Module): 归一化层
        """
        super(VisionTransformer, self).__init__()  # 调用父类构造函数
        # self.num_classes = num_classes  # 设置分类的类别数
        self.num_features = self.embed_dim = embed_dim  # 将嵌入维度赋值给 num_features，以保持一致性
        # self.num_tokens = 2 if distilled else 1  # 如果使用蒸馏，则设置 token 数量为 2；否则为 1

        # 设置归一化层，如果未指定，则使用 LayerNorm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # 使用偏函数来设置默认参数
        act_layer = act_layer or nn.GELU  # 如果未指定，则使用 GELU 作为激活函数

        # 创建图像块嵌入层
        self.patch_embed = embed_layer(image_size=image_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # 获取图像块的数量

        # 定义分类 token 和位置嵌入参数
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 类别 token，用于分类任务
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  # 蒸馏 token（可选）
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # 位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , embed_dim)) 
        self.pos_drop = nn.Dropout(p=drop_ratio)  # 位置嵌入的 dropout 层

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # 生成深度衰减值
        self.blocks = nn.Sequential(*[  # 创建多个 Transformer 块
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)  # 根据深度构建多个块
        ])
        self.norm = norm_layer(embed_dim)  # 最后的归一化层

        # 解码器部分
        self.decoder = Decoder(output_c=output_channel)

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # 初始化位置嵌入
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)  # 初始化蒸馏 token
        # nn.init.trunc_normal_(self.cls_token, std=0.02)  # 初始化分类 token
        self.apply(self._init_vit_weights)  # 应用自定义的初始化函数

    def forward_features(self, x):
        # 处理输入特征
        x = self.patch_embed(x)  # 将输入图像进行嵌入，输出形状为 [B, num_patches, embed_dim]
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 扩展 CLS token 以匹配批次大小
        # if self.dist_token is None:
        #     x = torch.cat((cls_token, x), dim=1)  # 将 CLS token 与嵌入的图像块连接
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)  # 如果有蒸馏 token，则连接它

        x = self.pos_drop(x + self.pos_embed)  # 添加位置嵌入并应用 dropout
        x = self.blocks(x)  # 通过多个 Transformer 块
        x = self.norm(x)  # 应用归一化
        return x[:, 0]  # 返回 CLS token 的输出作为特征

    def forward(self, x):
        x = self.forward_features(x)  # 处理输入图像，获取 CLS token 输出
        # print("x:",x.shape)
        # 解码器部分
        x = self.decoder(x)  # 通过解码器生成补全的图像
        return x  # 返回补全的图像


    def _init_vit_weights(self,m):
        """
        ViT weight initialization
        :param m: module
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
def ocean_model(output_channel):
 
    model = VisionTransformer(image_size=21,
                              patch_size=3,
                              in_c=5,
                              embed_dim=36,
                              depth=5,
                              num_heads=3,output_channel=output_channel
                              )
    #token叠在一起（49，36）

    return model

def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(image_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)

    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(image_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)

    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(image_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)

    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(image_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)

    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(image_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)

    return model
