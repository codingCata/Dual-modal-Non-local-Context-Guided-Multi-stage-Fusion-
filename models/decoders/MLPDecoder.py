import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from torch.nn.modules import module
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from einops import repeat


class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_heads=[8, 8, 8, 8],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=512,  # 768
                 align_corners=False):

        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        embedding_dim = embed_dim

        self.linear_fuse = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.guide = nn.ModuleList([
            Guide(in_channel=in_channels[0], out_channel=in_channels[3], reduction=1),
            Guide(in_channel=in_channels[1], out_channel=in_channels[3], reduction=1),
            Guide(in_channel=in_channels[2], out_channel=in_channels[3], reduction=1)])

        self.Fusion = nn.ModuleList([
            FusionModule(dim=in_channels[0], reduction=1, num_heads=num_heads[0]),
            FusionModule(dim=in_channels[1], reduction=1, num_heads=num_heads[1]),
            FusionModule(dim=in_channels[2], reduction=1, num_heads=num_heads[2]),
            FusionModule(dim=in_channels[3], reduction=1, num_heads=num_heads[3])])

    def forward(self, rgb, extra):
        f1 = self.Fusion[0](rgb[0], extra[0])
        f2 = self.Fusion[1](rgb[1], extra[1])
        f3 = self.Fusion[2](rgb[2], extra[2])
        f4 = self.Fusion[3](rgb[3], extra[3])

        f1 = self.guide[0](f1, f4)
        f2 = self.guide[1](f2, f4)
        f3 = self.guide[2](f3, f4)

        f4 = F.interpolate(f4, size=f1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        f3 = F.interpolate(f3, size=f1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        f2 = F.interpolate(f2, size=f1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        f = self.linear_fuse(torch.cat([f4, f3, f2, f1], dim=1))
        x = self.dropout(f)
        x = self.linear_pred(x)

        return x


class Guide(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=1, proj_drop=0., attn_drop=0., qkv_bias=False):
        super(Guide, self).__init__()
        self.q = nn.Linear(in_channel, out_channel // reduction, bias=qkv_bias)
        self.k = nn.Linear(out_channel, out_channel // reduction, bias=qkv_bias)
        self.v = nn.Linear(out_channel, out_channel // reduction, bias=qkv_bias)
        self.scale = (out_channel // reduction) ** -0.5
        self.proj = nn.Linear(out_channel // reduction, out_channel)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(in_channel, out_channel)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C1, H, W = x1.shape
        _, C2, _, _ = x2.shape
        x1 = x1.reshape(B, C1, -1).permute(0, 2, 1)
        x2 = x2.reshape(B, C2, -1).permute(0, 2, 1)
        q = self.q(x1)  # B N C
        k = self.k(x2)  # B n C
        v = self.v(x2)  # B n C

        attn = (q @ k.transpose(1, 2)) * self.scale  # N*C @ C*n
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # N*n @ n*C = N*C
        x = self.proj(x)
        x = self.proj_drop(x)
        x1 = self.proj1(x1)
        x = x + x1
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()  # B head N C/head
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale  # C*N * N*C -> C*C
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)  # B N C
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)  # B C H W

        return merge