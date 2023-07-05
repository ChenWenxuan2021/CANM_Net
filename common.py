import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange
import torch.nn.functional as F

import warnings 
warnings.filterwarnings('ignore')

def _to_channel_last(x):
    # N,C,H,W -> N,H,W,C
    
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    # N,H,W,C -> N,C,H,W
    
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size=(8,8), num_windows=(8,8)):
    # return local windows feature
    
    N, H, W, C = x.shape
    w1, w2 = window_size
    nw1, nw2 = num_windows
    x = x.view(N, nw1, w1, nw2, w2, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, (w1*w2), C)
    return windows


def window_reverse(windows, window_size=(8,8), num_windows=(8,8)):
    # return original form of 4D tensors
    
    w1, w2 = window_size
    nw1, nw2 = num_windows
    i1, i2 = w1*nw1, w2*nw2
    N = int(windows.shape[0] / (nw1 * nw2))
    x = windows.view(N, nw1, nw2, w1, w2, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(N, i1, i2, -1)
    return x


def compute_mask(H, W, window_size, shift_size, device):
    # compute mask for shifted-windows
    img_mask = torch.zeros((1, H, W, 1), device=device)
    cnt = 0
    for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            img_mask[:, h, w, :] = cnt
            cnt += 1
    num_windows = (H//window_size[0], W//window_size[1])
    mask_windows = window_partition(img_mask, window_size, num_windows)  # nW, ws[0]*ws[1], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class conv_in_relu2d(nn.Module):
    
    def __init__(self, in_chs, out_chs, kernel_size, stride, padding, 
                 norm=nn.InstanceNorm2d,
                 act=nn.LeakyReLU):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            norm(out_chs),
            act(0.2, inplace=True) if act is nn.LeakyReLU else act(inplace=True)
            )
    def forward(self, x):
        return self.model(x)


class WindowAttention(nn.Module):
    
    def __init__(self, dim, num_heads, window_size=(8,8), qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchMerging(nn.Module):
    # downsample for a pyramid structure
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        N, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1) # N, H/2, W/2, 4C
        x = self.norm(x)
        x = self.reduction(x)

        return x


class ChannelAttention_Pyramid(nn.Module):
    # channel-based attention
    
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature_lv1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_lv2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.weight_lv1 = nn.Parameter(torch.Tensor([0.5,]))
        self.weight_lv2 = nn.Parameter(torch.Tensor([0.5,]))
        
        self.down_lv1 = nn.Conv2d(dim, dim, 3, 2, 1, groups=dim, bias=bias)
        self.down_lv2 = nn.Conv2d(dim, dim, 4, 4, 0, groups=dim, bias=bias)
        
        self.to_qk_lv1 = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1, 1, 0, bias=bias),
            nn.Conv2d(dim*2, dim*2, 3, 1, 1, groups=dim, bias=bias)
            )
        self.to_qk_lv2 = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1, 1, 0, bias=bias),
            nn.Conv2d(dim*2, dim*2, 3, 1, 1, groups=dim, bias=bias)
            )
        self.to_v = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=bias),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
            )
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_lv1 = self.down_lv1(x)
        x_lv2 = self.down_lv2(x)
        
        q_lv1, k_lv1 = self.to_qk_lv1(x_lv1).chunk(2, dim=1)
        q_lv2, k_lv2 = self.to_qk_lv2(x_lv2).chunk(2, dim=1)
        
        q_lv1 = rearrange(q_lv1, 'B (head C) H W -> B head C (H W)', head=self.num_heads)
        k_lv1 = rearrange(k_lv1, 'B (head C) H W -> B head C (H W)', head=self.num_heads)
        q_lv2 = rearrange(q_lv2, 'B (head C) H W -> B head C (H W)', head=self.num_heads)
        k_lv2 = rearrange(k_lv2, 'B (head C) H W -> B head C (H W)', head=self.num_heads)
        
        q_lv1 = F.normalize(q_lv1, dim=-1)
        k_lv1 = F.normalize(k_lv1, dim=-1)
        q_lv2 = F.normalize(q_lv2, dim=-1)
        k_lv2 = F.normalize(k_lv2, dim=-1)
        
        attn_lv1 = (q_lv1 @ k_lv1.transpose(-2, -1)) * self.temperature_lv1
        attn_lv2 = (q_lv2 @ k_lv2.transpose(-2, -1)) * self.temperature_lv2
        attn = attn_lv1 * self.weight_lv1 + attn_lv2 * self.weight_lv2
        attn = attn.softmax(dim=-1)
        
        v = self.to_v(x)
        v = rearrange(v, 'B (head C) H W -> B head C (H W)', head=self.num_heads)
        out = (attn @ v)
        out = rearrange(out, 'B head C (H W) -> B (head C) H W', head=self.num_heads, H=H, W=W)
        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    # using feedforward by conv2d, instead of mlp
    
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SE(nn.Module):
    # Squeeze and excitation block
    
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class WithBias_LayerNorm(nn.Module):
    # using the version of Restormer

    def __init__(self, dim):
        super().__init__()
        normalized_shape = torch.Size((dim,))
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def _to_3d(self, x):
        return rearrange(x, 'b c h w -> b (h w) c')
    
    def _to_4d(self, x, h, w):
        return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
    
    def forward(self, x):
        H, W = x.shape[-2:]
        x_3d = self._to_3d(x)
        mu = x_3d.mean(-1, keepdim=True)
        sigma = x_3d.var(-1, keepdim=True, unbiased=False)
        x_3d = (x_3d - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        x = self._to_4d(x_3d, H, W)
        return x


class Downsample(nn.Module):
    # downsample using pixel-unshuffle
    
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim//2, 3, 1, 1, bias=False),
                                  nn.PixelUnshuffle(2))
    
    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    # upsample usint pixel-shuffle
    
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim*2, 3, 1, 1, bias=False),
                                  nn.PixelShuffle(2))
    
    def forward(self, x):
        return self.body(x)


class PatchEmbed(nn.Module):
    # patch-embed module for input of transformer
    
    def __init__(self, in_dim, dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, dim//2, 3, 1, 1)
        self.embed = nn.Sequential(
            nn.Conv2d(dim//2, dim//2, 3, 1, 1, groups=dim//2, bias=False),
            nn.GELU(),
            SE(dim//2, dim//2),
            nn.Conv2d(dim//2, dim//2, 1, 1, 0, bias=False),
            )
        self.reduction = nn.Conv2d(dim//2, dim, 3, 2, 1)
        self.norm1 = nn.LayerNorm(dim//2)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = self.conv(x)
        x1 = x
        x = _to_channel_last(x).contiguous()
        x = self.norm1(x)
        x = _to_channel_first(x)
        x = x + self.embed(x)
        x = self.reduction(x)
        x = _to_channel_last(x)
        x = self.norm2(x)
        x = _to_channel_first(x)
        return x, x1