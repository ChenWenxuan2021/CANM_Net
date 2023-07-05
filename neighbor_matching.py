import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AddStyle(nn.Module):
    # adding the mean and std. of degraded feature to that of ref features
    
    def __init__(self, dim):
        super().__init__()
        self.norm_ref = nn.InstanceNorm2d(dim, affine=False)
        self.conv_shared = nn.Sequential(nn.Conv2d(dim*2, dim, 3, 1, 1),
                                         nn.ReLU(inplace=True))
        self.conv_beta = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv_gamma = nn.Conv2d(dim, dim, 3, 1, 1)
        
        self.conv_gamma.weight.data.zero_()
        self.conv_beta.weight.data.zero_()
        self.conv_gamma.bias.data.zero_()
        self.conv_beta.bias.data.zero_()
        
    def forward(self, degraded, ref):
        B, C, H, W = degraded.shape
        ref_normed = self.norm_ref(ref)
        
        style = self.conv_shared(torch.cat([degraded, ref],dim=1))
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)
        
        degraded = degraded.view(B, C, H*W)
        mu = torch.mean(degraded, dim=-1, keepdim=True).unsqueeze(3)
        sigma = torch.std(degraded, dim=-1, keepdim=True).unsqueeze(3)
        
        gamma = gamma + sigma
        beta = beta + mu
        
        ref_styled = ref_normed * gamma + beta
        return ref_styled


class NBFM(nn.Module):
    # neighborhood-based feature matching
    
    def __init__(self,
                 dim, input_resolution,
                 patch_size, patch_dilation, patch_stride,
                 neighbor_size,
                 gamma = 6,
                 ):
        super().__init__()
        self.resolution = input_resolution
        self.patch_size = patch_size
        self.patch_dilation = patch_dilation
        self.patch_stride = patch_stride
        self.neighbor_size = neighbor_size
        self.gamma = gamma
        
        self._make_learnable_weight()
        self._make_neighbor_list()
        
        # addstyle
        self.addstyle_before = AddStyle(dim)
        self.addstyle_after = AddStyle(dim)
        
        # unfold degraded and ref to patch
        padding_degraded = (patch_size + (patch_dilation - 1) * 2 - 1) // 2
        padding_ref = padding_degraded + ((neighbor_size - 1) // 2) * patch_stride
        self.ufd_degraded = nn.Unfold(patch_size, patch_dilation, padding_degraded, patch_stride)
        self.ufd_ref = nn.Unfold(patch_size, patch_dilation, padding_ref, patch_stride)
        
        # fold
        self.fold = nn.Fold(input_resolution, patch_size, patch_dilation, padding_degraded, patch_stride)
        
        # convolution feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1, 1, 0,),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.ReLU(inplace=True),
            )
    
    def _make_learnable_weight(self):
        X = np.linspace(0, self.neighbor_size-1, self.neighbor_size,dtype=np.float32)
        Y = np.linspace(0, self.neighbor_size-1, self.neighbor_size,dtype=np.float32)
        x, y = np.meshgrid(X, Y)
        mean = (self.neighbor_size-1) // 2
        weight = np.exp(-((x-mean)**2 + (y-mean)**2) / self.gamma)
        weight = torch.from_numpy(weight).view(-1)
        self.weight = nn.Parameter(weight)
    
    def _make_neighbor_list(self):
        degraded_len = self.resolution[0] // self.patch_stride # adaptive padding to avoid change size
        ref_len = degraded_len + self.neighbor_size - 1
        neighbor_list = np.zeros([degraded_len**2, self.neighbor_size**2], dtype=np.int32)
        
        for i in range(neighbor_list.shape[0]): # i: determine the degraded position
            degraded_x = i % degraded_len
            degraded_y = i // degraded_len
            for j in range(neighbor_list.shape[1]): # j: determine the relative-position between degraded and ref
                delta_x = j % self.neighbor_size
                delta_y = j // self.neighbor_size
                center_x = degraded_x + delta_x
                center_y = degraded_y + delta_y
                i_ref = center_x + center_y * (ref_len)
                neighbor_list[i, j] = i_ref
        self.neighbor_list = neighbor_list
    
    def forward(self, degraded, ref):
        ref = self.addstyle_before(degraded, ref)
        
        degraded_patches = self.ufd_degraded(degraded)
        ref_patches = self.ufd_ref(ref)
        degraded_patches = F.normalize(degraded_patches, dim=-2)
        ref_patches = F.normalize(ref_patches, dim=-2)
        
        # group-based matrix-multiplication (neighborhood-based cosine similarity)
        ref_patches_grouped = ref_patches[:, :, self.neighbor_list]
        ref_patches_grouped = ref_patches_grouped.transpose(1, 2)
        degraded_patches = degraded_patches.transpose(-1, -2).unsqueeze(-2)
        attn = torch.matmul(degraded_patches, ref_patches_grouped)
        attn = attn * self.weight # using learnable weight to control overall style
        attn = attn.softmax(dim=-1)
        matched_patches = (ref_patches_grouped * attn).sum(dim=-1)
        
        # fold, and texture transfer
        matched = self.fold(matched_patches.transpose(-2, -1))
        matched = self.addstyle_after(degraded, matched)
        fused = self.fusion(torch.cat([degraded, matched], dim=1))
        return fused
        

if __name__ == '__main__':
    degraded = torch.rand((8, 64, 128, 128))
    ref = torch.rand((8, 64, 128, 128))
    module = NBFM(dim=64, input_resolution=128, patch_size=5, patch_dilation=1, 
                  patch_stride=2, neighbor_size=3, gamma=6)
    degraded = degraded.cuda()
    ref = ref.cuda()
    module = module.cuda()
    fused = module(degraded, ref)
    