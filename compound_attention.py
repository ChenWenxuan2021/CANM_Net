import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
import common
from einops import rearrange


class CTL(nn.Module):
    # compound transformer layer
    
    def __init__(self, dim, input_resolution, WA_heads, CA_heads, window_size=(8,8),
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 shift_size = None,
                 CA_bias=False, ffn_expansion_factor=2, FF_bias=False,
                 ):
        super().__init__()
        # WAB: window-based attention block
        self.window_size = window_size
        self.norm_wa = nn.LayerNorm(dim)
        self.wa = common.WindowAttention(dim,
                                         num_heads=WA_heads,
                                         window_size=window_size,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         attn_drop=attn_drop,
                                         proj_drop=drop,
                                         )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # CAB: channel-based attention block
        self.norm_ca = common.WithBias_LayerNorm(dim)
        self.ca = common.ChannelAttention_Pyramid(dim, num_heads=CA_heads, bias=CA_bias)
        
        # FFB: feed-forward block
        self.reduce_dim = nn.Conv2d(2*dim, dim, 1, 1, 0)
        self.norm_ff = common.WithBias_LayerNorm(dim)
        self.ff = common.FeedForward(dim, ffn_expansion_factor, bias=FF_bias)
        
        # helper function
        self.window_size = window_size
        i1, i2 = input_resolution
        w1, w2 = window_size
        nw1, nw2 = i1 // w1, i2 // w2
        self.num_windows = (nw1, nw2)
        self.shift_size = shift_size
        if shift_size is not None:
            self.shift_size = shift_size
            self.using_shift = True
        else:
            self.using_shift = False
        
    def forward(self, x, attn_mask):
        B, H, W, C = x.shape
        shortcut = x
        
        # WAB
        x_for_wa = common._to_channel_last(x)
        x_for_wa = self.norm_wa(x_for_wa)
        
        if self.using_shift:
            shift_size = self.shift_size
            x_for_wa_shifted = torch.roll(x_for_wa, shifts=(-shift_size[0], -shift_size[1]),
                                   dims=(1, 2))
        else:
            x_for_wa_shifted = x_for_wa
            attn_mask = None
            
        x_windows = common.window_partition(x_for_wa_shifted, self.window_size, self.num_windows)
        attn_windows = self.wa(x_windows, attn_mask)
        x_for_wa_shifted = common.window_reverse(attn_windows, self.window_size, self.num_windows)
        
        if self.using_shift:
            x_for_wa = torch.roll(x_for_wa_shifted, shifts=(shift_size[0], shift_size[1]),
                                   dims=(1, 2))
        else:
            x_for_wa = x_for_wa_shifted
        
        x_for_wa = common._to_channel_first(x_for_wa)
        
        # CAB
        x_for_ca = self.norm_ca(x)
        x_for_ca = self.ca(x)
        
        # Reduce dim
        x = self.reduce_dim(torch.cat([x_for_wa, x_for_ca], dim=1))
        x = x + shortcut
        
        # FFB
        x = x + self.ff(self.norm_ff(x))
        
        return x


class CTE(nn.Module):
    # compound transformer encoder modules
    
    def __init__(self, dim, depth, input_resolution, WA_heads, CA_heads, window_size=(8,8),
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 CA_bias=False, ffn_expansion_factor=2, FF_bias=False,
                 reduce_size = True,
                 ):
        super().__init__()
        shift_size = (window_size[0]//2, window_size[1]//2)
        self.shift_size = shift_size
        self.window_size = window_size

        self.body = nn.ModuleList([])
        self.depth = depth
        for i in range(depth):
            self.body.append(
                CTL(dim, input_resolution, WA_heads, CA_heads, window_size, qkv_bias,
                    qk_scale, drop, attn_drop, drop_path, shift_size if i % 2 == 1 else None,
                    CA_bias, ffn_expansion_factor, FF_bias)
                )
            
        self.reduce_size = reduce_size
        if reduce_size:
            self.downsample = common.Downsample(dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        attn_mask = common.compute_mask(Hp, Wp, self.window_size, self.shift_size, x.device)
        
        for i in range(self.depth):
            x = self.body[i](x, attn_mask)
        
        if self.reduce_size:
            x_ds = self.downsample(x)
            return x, x_ds
        return x, None


class CTD(nn.Module):
    # compound transformer decoder modules
    
    def __init__(self, dim, depth, input_resolution, WA_heads, CA_heads, window_size=(8,8),
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 CA_bias=False, ffn_expansion_factor=2, FF_bias=False,
                 increase_size = True,
                 ):
        super().__init__()
        shift_size = (window_size[0]//2, window_size[1]//2)
        self.shift_size = shift_size
        self.window_size = window_size

        self.body = nn.ModuleList([])
        self.depth = depth
        for i in range(depth):
            self.body.append(
                CTL(dim, input_resolution, WA_heads, CA_heads, window_size, qkv_bias,
                    qk_scale, drop, attn_drop, drop_path, shift_size if i % 2 == 1 else None,
                    CA_bias, ffn_expansion_factor, FF_bias)
                )
            
        self.increase_size = increase_size
        if increase_size:
            self.upsample = common.Upsample(dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        attn_mask = common.compute_mask(Hp, Wp, self.window_size, self.shift_size, x.device)
        
        for i in range(self.depth):
            x = self.body[i](x, attn_mask)
            
        if self.increase_size:
            x = self.upsample(x)
        return x


class Layer_WAonly(nn.Module):
    
    def __init__(self, dim, input_resolution, WA_heads, CA_heads, window_size=(8,8),
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 shift_size = None,
                 CA_bias=False, ffn_expansion_factor=2, FF_bias=False,
                 ):
        super().__init__()
        self.window_size = window_size
        self.norm_wa = nn.LayerNorm(dim)
        self.wa = common.WindowAttention(dim,
                                          num_heads=WA_heads,
                                          window_size=window_size,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,
                                          attn_drop=attn_drop,
                                          proj_drop=drop,
                                          )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_ff = common.WithBias_LayerNorm(dim)
        self.ff = common.FeedForward(dim, ffn_expansion_factor, bias=FF_bias)
        self.window_size = window_size
        i1, i2 = input_resolution
        w1, w2 = window_size
        nw1, nw2 = i1 // w1, i2 // w2
        self.num_windows = (nw1, nw2)
        self.shift_size = shift_size
        if shift_size is not None:
            self.shift_size = shift_size
            self.using_shift = True
        else:
            self.using_shift = False

    def forward(self, x, attn_mask):
        B, H, W, C = x.shape
        shortcut = x
        
        # WAB
        x_for_wa = common._to_channel_last(x)
        x_for_wa = self.norm_wa(x_for_wa)
        
        if self.using_shift:
            shift_size = self.shift_size
            x_for_wa_shifted = torch.roll(x_for_wa, shifts=(-shift_size[0], -shift_size[1]),
                                    dims=(1, 2))
        else:
            x_for_wa_shifted = x_for_wa
            attn_mask = None
            
        x_windows = common.window_partition(x_for_wa_shifted, self.window_size, self.num_windows)
        attn_windows = self.wa(x_windows, attn_mask)
        x_for_wa_shifted = common.window_reverse(attn_windows, self.window_size, self.num_windows)
        
        if self.using_shift:
            x_for_wa = torch.roll(x_for_wa_shifted, shifts=(shift_size[0], shift_size[1]),
                                    dims=(1, 2))
        else:
            x_for_wa = x_for_wa_shifted
        
        x_for_wa = common._to_channel_first(x_for_wa)
        x = x_for_wa + shortcut
        
        # FFB
        x = x + self.ff(self.norm_ff(x))
        return x


class Layer_CAonly(nn.Module):
    
    def __init__(self, dim, input_resolution, WA_heads, CA_heads, window_size=(8,8),
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 shift_size = None,
                 CA_bias=False, ffn_expansion_factor=2, FF_bias=False,
                 ):
        super().__init__()
        self.norm_ca = common.WithBias_LayerNorm(dim)
        self.ca = common.ChannelAttention_Pyramid(dim, num_heads=CA_heads, bias=CA_bias)
        self.norm_ff = common.WithBias_LayerNorm(dim)
        self.ff = common.FeedForward(dim, ffn_expansion_factor, bias=FF_bias)

    def forward(self, x, attn_mask):
        B, H, W, C = x.shape
        shortcut = x
        
        # CAB
        x_for_ca = self.norm_ca(x)
        x_for_ca = self.ca(x)
        x = shortcut + x_for_ca
        
        # FFB
        x = x + self.ff(self.norm_ff(x))
        return x


class ChannelAttention_noPyramid(nn.Module):
    # channel-based attention
    
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class CTL_noPS(nn.Module):
    # compound transformer layer
    
    def __init__(self, dim, input_resolution, WA_heads, CA_heads, window_size=(8,8),
                  qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                  shift_size = None,
                  CA_bias=False, ffn_expansion_factor=2, FF_bias=False,
                  ):
        super().__init__()
        # WAB: window-based attention block
        self.window_size = window_size
        self.norm_wa = nn.LayerNorm(dim)
        self.wa = common.WindowAttention(dim,
                                          num_heads=WA_heads,
                                          window_size=window_size,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,
                                          attn_drop=attn_drop,
                                          proj_drop=drop,
                                          )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # CAB: channel-based attention block
        self.norm_ca = common.WithBias_LayerNorm(dim)
        self.ca = ChannelAttention_noPyramid(dim, num_heads=CA_heads, bias=CA_bias)
        
        # FFB: feed-forward block
        self.reduce_dim = nn.Conv2d(2*dim, dim, 1, 1, 0)
        self.norm_ff = common.WithBias_LayerNorm(dim)
        self.ff = common.FeedForward(dim, ffn_expansion_factor, bias=FF_bias)
        
        # helper function
        self.window_size = window_size
        i1, i2 = input_resolution
        w1, w2 = window_size
        nw1, nw2 = i1 // w1, i2 // w2
        self.num_windows = (nw1, nw2)
        self.shift_size = shift_size
        if shift_size is not None:
            self.shift_size = shift_size
            self.using_shift = True
        else:
            self.using_shift = False
        
    def forward(self, x, attn_mask):
        B, H, W, C = x.shape
        shortcut = x
        
        # WAB
        x_for_wa = common._to_channel_last(x)
        x_for_wa = self.norm_wa(x_for_wa)
        
        if self.using_shift:
            shift_size = self.shift_size
            x_for_wa_shifted = torch.roll(x_for_wa, shifts=(-shift_size[0], -shift_size[1]),
                                    dims=(1, 2))
        else:
            x_for_wa_shifted = x_for_wa
            attn_mask = None
            
        x_windows = common.window_partition(x_for_wa_shifted, self.window_size, self.num_windows)
        attn_windows = self.wa(x_windows, attn_mask)
        x_for_wa_shifted = common.window_reverse(attn_windows, self.window_size, self.num_windows)
        
        if self.using_shift:
            x_for_wa = torch.roll(x_for_wa_shifted, shifts=(shift_size[0], shift_size[1]),
                                    dims=(1, 2))
        else:
            x_for_wa = x_for_wa_shifted
        
        x_for_wa = common._to_channel_first(x_for_wa)
        
        # CAB
        x_for_ca = self.norm_ca(x)
        x_for_ca = self.ca(x)
        
        # Reduce dim
        x = self.reduce_dim(torch.cat([x_for_wa, x_for_ca], dim=1))
        x = x + shortcut
        
        # FFB
        x = x + self.ff(self.norm_ff(x))
        
        return x


class CTE_ablation(nn.Module):
    # compound transformer encoder modules
    
    def __init__(self, dim, depth, input_resolution, WA_heads, CA_heads, window_size=(8,8),
                  qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                  CA_bias=False, ffn_expansion_factor=2, FF_bias=False,
                  reduce_size = True,
                  Layer = None, # Layer_WAonly? or Layer_CAonly? or CTL_noPS?
                  ):
        super().__init__()
        shift_size = (window_size[0]//2, window_size[1]//2)
        self.shift_size = shift_size
        self.window_size = window_size

        self.body = nn.ModuleList([])
        self.depth = depth
        for i in range(depth):
            self.body.append(
                Layer(dim, input_resolution, WA_heads, CA_heads, window_size, qkv_bias,
                    qk_scale, drop, attn_drop, drop_path, shift_size if i % 2 == 1 else None,
                    CA_bias, ffn_expansion_factor, FF_bias)
                )
            
        self.reduce_size = reduce_size
        if reduce_size:
            self.downsample = common.Downsample(dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        attn_mask = common.compute_mask(Hp, Wp, self.window_size, self.shift_size, x.device)
        
        for i in range(self.depth):
            x = self.body[i](x, attn_mask)
        
        if self.reduce_size:
            x_ds = self.downsample(x)
            return x, x_ds
        return x, None


class CTD_ablation(nn.Module):
    # compound transformer decoder modules
    
    def __init__(self, dim, depth, input_resolution, WA_heads, CA_heads, window_size=(8,8),
                  qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                  CA_bias=False, ffn_expansion_factor=2, FF_bias=False,
                  increase_size = True,
                  Layer = None, # Layer_WAonly? or Layer_CAonly? or CTL_noPS?
                  ):
        super().__init__()
        shift_size = (window_size[0]//2, window_size[1]//2)
        self.shift_size = shift_size
        self.window_size = window_size

        self.body = nn.ModuleList([])
        self.depth = depth
        for i in range(depth):
            self.body.append(
                Layer(dim, input_resolution, WA_heads, CA_heads, window_size, qkv_bias,
                    qk_scale, drop, attn_drop, drop_path, shift_size if i % 2 == 1 else None,
                    CA_bias, ffn_expansion_factor, FF_bias)
                )
            
        self.increase_size = increase_size
        if increase_size:
            self.upsample = common.Upsample(dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        attn_mask = common.compute_mask(Hp, Wp, self.window_size, self.shift_size, x.device)
        
        for i in range(self.depth):
            x = self.body[i](x, attn_mask)
            
        if self.increase_size:
            x = self.upsample(x)
        return x