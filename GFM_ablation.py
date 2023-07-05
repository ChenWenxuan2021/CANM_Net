import torch
import torch.nn as nn

import common
import compound_attention
# import neighbor_matching
import torch.nn.functional as F


class SearchTransfer(nn.Module):
    
    def __init__(self, dim, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.fusion = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1, 1, 0,),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.ReLU(inplace=True),
            )
        
    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)
    
    def forward(self, lr, ref):
        B, C, H, W = ref.shape
        ref_unfold = F.unfold(ref, kernel_size=(3,3), stride=self.stride, padding=self.padding)
        lr_unfold = F.unfold(lr, kernel_size=(3,3), stride=self.stride, padding=self.padding)
        ref_unfold = ref_unfold.permute(0, 2, 1)
        ref_unfold = F.normalize(ref_unfold, dim=2)
        lr_unfold = F.normalize(lr_unfold, dim=1)
        R = torch.bmm(ref_unfold, lr_unfold)
        R_star, R_star_arg = torch.max(R, dim=1)
        
        ref_unfold = ref_unfold.permute(0, 2, 1)
        T_unfold = self.bis(ref_unfold, 2, R_star_arg)
        T = F.fold(T_unfold, (H,W), kernel_size=(3,3), stride=self.stride, padding=self.padding)
        fused = self.fusion(torch.cat([lr, T],dim=1))
        return fused


class ReduceDim(nn.Module):
    # depth-wise conv to reduce dimensional
    
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim//2, 1, 1, 0,),
            nn.Conv2d(dim//2, dim//2, 3, 1, 1, bias=False)
            )
    
    def forward(self, x):
        return self.body(x)


class GFM(nn.Module):
    # a pytorch implement of the proposed CANM-Net's architecture
    
    def __init__(self,
                 dim=64, # channel dimension in the first level
                 # for compound attention
                 CTE_depth = [4, 4, 16, 4], # depth of encoder modules
                 CTD_depth = [4, 4, 4], # depth of decoder modules
                 WA_heads = [2, 4, 8, 8], # window attention number of heads
                 CA_heads = [1, 2, 4, 8], # channel attention number of heads
                 embed_resolution = (128, 128), # input resolution in the first level
                 window_size=(8, 8), # for window attention
                 CA_bias = False, # for bias-free of with-bias channel attention
                 FF_bias = False, # for bias-free of with-bias gated-Dconv feed forward
                 ffn_expansion_factor = 2,
                 # for neighbor matching
                 patch_size = [3, 3, 3], # patch size for 1st to 3rd NBFM modules
                 patch_dilation = [1, 2, 1],
                 patch_stride = [2, 1, 1],
                 neighbor_size = [3, 3, 5],
                 gamma = [6, 6, 6],
                 ):
        super().__init__()
        self.patch_embed_degraded = common.PatchEmbed(2, dim)
        self.patch_embed_ref = common.PatchEmbed(2, dim)
        
        self.num_encoders = len(CTE_depth)
        self.num_decoders = len(CTD_depth)
        
        # encoder
        self.encoder_degraded = nn.ModuleList([])
        self.encoder_ref = nn.ModuleList([])
        for i in range(len(CTE_depth)):
            factor = 2 ** i
            dim_level = dim * factor
            resolution_level = (embed_resolution[0] // factor, embed_resolution[1] // factor)
            self.encoder_degraded.append(
                compound_attention.CTE(dim=dim_level, depth=CTE_depth[i],
                                       input_resolution=resolution_level, WA_heads=WA_heads[i],
                                       CA_heads=CA_heads[i], window_size=window_size,
                                       CA_bias=CA_bias, FF_bias=FF_bias,
                                       ffn_expansion_factor=ffn_expansion_factor,
                                       reduce_size=True if i < len(CTE_depth)-1 else False,
                                       )
                )
            self.encoder_ref.append(
                compound_attention.CTE(dim=dim_level, depth=CTE_depth[i],
                                       input_resolution=resolution_level, WA_heads=WA_heads[i],
                                       CA_heads=CA_heads[i], window_size=window_size,
                                       CA_bias=CA_bias, FF_bias=FF_bias,
                                       ffn_expansion_factor=ffn_expansion_factor,
                                       reduce_size=True if i < len(CTE_depth)-1 else False,
                                       )
                )
        
        ## TODO
        # feature matching
        # self.nbfm_lv1 = neighbor_matching.NBFM(dim, (embed_resolution[0], embed_resolution[1]),
        #                                        patch_size[0], patch_dilation[0], patch_stride[0],
        #                                        neighbor_size[0], gamma[0],
        #                                        )
        # self.nbfm_lv2 = neighbor_matching.NBFM(dim*2, (embed_resolution[0]//2, embed_resolution[1]//2),
        #                                        patch_size[1], patch_dilation[1], patch_stride[1],
        #                                        neighbor_size[1], gamma[1],
        #                                        )
        # self.nbfm_lv3 = neighbor_matching.NBFM(dim*4, (embed_resolution[0]//4, embed_resolution[1]//4),
        #                                        patch_size[2], patch_dilation[2], patch_stride[2],
        #                                        neighbor_size[2], gamma[2],
        #                                        )
        
        self.gfm_lv1 = SearchTransfer(dim, stride=2, padding=1)
        self.gfm_lv2 = SearchTransfer(dim*2, stride=1, padding=1)
        self.gfm_lv3 = SearchTransfer(dim*4, stride=1, padding=1)
        
        
        # decoder
        self.reduce_dim_lv4 = nn.Sequential(ReduceDim(dim*16), nn.ReLU(inplace=True),)
        self.upsample_lv4 = common.Upsample(dim*8)
        
        self.reduce_lv1to3 = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        for i in range(len(CTD_depth)):
            factor = 2 ** i
            dim_level = dim * factor
            resolution_level = (embed_resolution[0] // factor, embed_resolution[1] // factor)
            self.reduce_lv1to3.append(ReduceDim(dim*factor*2))
            self.decoders.append(
                compound_attention.CTD(dim=dim_level, depth=CTD_depth[i],
                                       input_resolution=resolution_level, WA_heads=WA_heads[i],
                                       CA_heads=CA_heads[i],window_size=window_size,
                                       CA_bias=CA_bias, FF_bias=FF_bias,
                                       ffn_expansion_factor=ffn_expansion_factor,
                                       increase_size=True if i > 0 else False,
                                       )
                )
        
        self.upsample_lv1 = common.Upsample(dim)
        self.to_out = nn.Conv2d(dim, 1, 3, 1, 1)
    
    def forward(self, x):
        degraded_feature_input, shortcut = self.patch_embed_degraded(x)
        ref_feature_input, _ = self.patch_embed_ref(x)
        degraded_feature_list = []
        ref_feature_list = []
        
        # feature extraction
        for i in range(self.num_encoders):
            degraded_feature, degraded_feature_input = self.encoder_degraded[i](degraded_feature_input)
            degraded_feature_list.append(degraded_feature)
            ref_feature, ref_feature_input = self.encoder_ref[i](ref_feature_input)
            ref_feature_list.append(ref_feature)
        
        fused_feature_list = []
        # feature matching
        fused_feature_list.append(self.gfm_lv1(degraded_feature_list[0], ref_feature_list[0]))
        fused_feature_list.append(self.gfm_lv2(degraded_feature_list[1], ref_feature_list[1]))
        fused_feature_list.append(self.gfm_lv3(degraded_feature_list[2], ref_feature_list[2]))
        
        # upsample
        feature = self.reduce_dim_lv4(torch.cat([degraded_feature_list[-1],ref_feature_list[-1]],dim=1))
        feature = self.upsample_lv4(feature)
        for i in range(self.num_decoders-1, -1, -1):
            feature = self.reduce_lv1to3[i](torch.cat([feature, fused_feature_list[i]], dim=1))
            feature = self.decoders[i](feature)
        feature = self.upsample_lv1(feature)
        output = self.to_out(torch.cat([feature, shortcut], dim=1))
        return output


class woFM(nn.Module):
    # a pytorch implement of the proposed CANM-Net's architecture
    
    def __init__(self,
                 dim=64, # channel dimension in the first level
                 # for compound attention
                 CTE_depth = [4, 4, 16, 4], # depth of encoder modules
                 CTD_depth = [4, 4, 4], # depth of decoder modules
                 WA_heads = [2, 4, 8, 8], # window attention number of heads
                 CA_heads = [1, 2, 4, 8], # channel attention number of heads
                 embed_resolution = (128, 128), # input resolution in the first level
                 window_size=(8, 8), # for window attention
                 CA_bias = False, # for bias-free of with-bias channel attention
                 FF_bias = False, # for bias-free of with-bias gated-Dconv feed forward
                 ffn_expansion_factor = 2,
                 # for neighbor matching
                 patch_size = [3, 3, 3], # patch size for 1st to 3rd NBFM modules
                 patch_dilation = [1, 2, 1],
                 patch_stride = [2, 1, 1],
                 neighbor_size = [3, 3, 5],
                 gamma = [6, 6, 6],
                 ):
        super().__init__()
        self.patch_embed_degraded = common.PatchEmbed(2, dim)
        self.patch_embed_ref = common.PatchEmbed(2, dim)
        
        self.num_encoders = len(CTE_depth)
        self.num_decoders = len(CTD_depth)
        
        # encoder
        self.encoder_degraded = nn.ModuleList([])
        self.encoder_ref = nn.ModuleList([])
        for i in range(len(CTE_depth)):
            factor = 2 ** i
            dim_level = dim * factor
            resolution_level = (embed_resolution[0] // factor, embed_resolution[1] // factor)
            self.encoder_degraded.append(
                compound_attention.CTE(dim=dim_level, depth=CTE_depth[i],
                                       input_resolution=resolution_level, WA_heads=WA_heads[i],
                                       CA_heads=CA_heads[i], window_size=window_size,
                                       CA_bias=CA_bias, FF_bias=FF_bias,
                                       ffn_expansion_factor=ffn_expansion_factor,
                                       reduce_size=True if i < len(CTE_depth)-1 else False,
                                       )
                )
            self.encoder_ref.append(
                compound_attention.CTE(dim=dim_level, depth=CTE_depth[i],
                                       input_resolution=resolution_level, WA_heads=WA_heads[i],
                                       CA_heads=CA_heads[i], window_size=window_size,
                                       CA_bias=CA_bias, FF_bias=FF_bias,
                                       ffn_expansion_factor=ffn_expansion_factor,
                                       reduce_size=True if i < len(CTE_depth)-1 else False,
                                       )
                )
        
        ## TODO
        # feature matching
        # self.nbfm_lv1 = neighbor_matching.NBFM(dim, (embed_resolution[0], embed_resolution[1]),
        #                                        patch_size[0], patch_dilation[0], patch_stride[0],
        #                                        neighbor_size[0], gamma[0],
        #                                        )
        # self.nbfm_lv2 = neighbor_matching.NBFM(dim*2, (embed_resolution[0]//2, embed_resolution[1]//2),
        #                                        patch_size[1], patch_dilation[1], patch_stride[1],
        #                                        neighbor_size[1], gamma[1],
        #                                        )
        # self.nbfm_lv3 = neighbor_matching.NBFM(dim*4, (embed_resolution[0]//4, embed_resolution[1]//4),
        #                                        patch_size[2], patch_dilation[2], patch_stride[2],
        #                                        neighbor_size[2], gamma[2],
        #                                        )       
        
        self.direct_concat_lv1 = ReduceDim(dim*2)
        self.direct_concat_lv2 = ReduceDim(dim*4)
        self.direct_concat_lv3 = ReduceDim(dim*8)
        
        # decoder
        self.reduce_dim_lv4 = nn.Sequential(ReduceDim(dim*16), nn.ReLU(inplace=True),)
        self.upsample_lv4 = common.Upsample(dim*8)
        
        self.reduce_lv1to3 = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        for i in range(len(CTD_depth)):
            factor = 2 ** i
            dim_level = dim * factor
            resolution_level = (embed_resolution[0] // factor, embed_resolution[1] // factor)
            self.reduce_lv1to3.append(ReduceDim(dim*factor*2))
            self.decoders.append(
                compound_attention.CTD(dim=dim_level, depth=CTD_depth[i],
                                       input_resolution=resolution_level, WA_heads=WA_heads[i],
                                       CA_heads=CA_heads[i],window_size=window_size,
                                       CA_bias=CA_bias, FF_bias=FF_bias,
                                       ffn_expansion_factor=ffn_expansion_factor,
                                       increase_size=True if i > 0 else False,
                                       )
                )
        
        self.upsample_lv1 = common.Upsample(dim)
        self.to_out = nn.Conv2d(dim, 1, 3, 1, 1)
    
    def forward(self, x):
        degraded_feature_input, shortcut = self.patch_embed_degraded(x)
        ref_feature_input, _ = self.patch_embed_ref(x)
        degraded_feature_list = []
        ref_feature_list = []
        
        # feature extraction
        for i in range(self.num_encoders):
            degraded_feature, degraded_feature_input = self.encoder_degraded[i](degraded_feature_input)
            degraded_feature_list.append(degraded_feature)
            ref_feature, ref_feature_input = self.encoder_ref[i](ref_feature_input)
            ref_feature_list.append(ref_feature)
        
        fused_feature_list = []
        # direct concat
        fused_feature_list.append(self.direct_concat_lv1(torch.cat([degraded_feature_list[0], ref_feature_list[0]],dim=1)))
        fused_feature_list.append(self.direct_concat_lv2(torch.cat([degraded_feature_list[1], ref_feature_list[1]],dim=1)))
        fused_feature_list.append(self.direct_concat_lv3(torch.cat([degraded_feature_list[2], ref_feature_list[2]],dim=1)))
        
        
        # upsample
        feature = self.reduce_dim_lv4(torch.cat([degraded_feature_list[-1],ref_feature_list[-1]],dim=1))
        feature = self.upsample_lv4(feature)
        for i in range(self.num_decoders-1, -1, -1):
            feature = self.reduce_lv1to3[i](torch.cat([feature, fused_feature_list[i]], dim=1))
            feature = self.decoders[i](feature)
        feature = self.upsample_lv1(feature)
        output = self.to_out(torch.cat([feature, shortcut], dim=1))
        return output


if __name__ == '__main__':
    degraded = torch.rand([4, 1, 256, 256])
    ref = torch.rand([4, 1, 256, 256])
    x = torch.cat([degraded, ref], dim=1)
    model = woFM()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    output = model(x)
    # from thop import profile
    # flops, params = profile(model=model, inputs=(x,))
    # params /= 1e6
    # flops /= 1e9