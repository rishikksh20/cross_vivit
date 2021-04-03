import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward, CrossAttention
import numpy as np



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self, spatial_dim = 96, spatial_depth = 4, spatial_heads =3, spatial_dim_head = 32, spatial_mlp_dim = 384,
                 temporal_dim = 192, temporal_depth = 1, temporal_heads = 3, temporal_dim_head = 64, temporal_mlp_dim = 768,
                 cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0.):
        super().__init__()
        self.transformer_enc_spatial = Transformer(spatial_dim, spatial_depth, spatial_heads, spatial_dim_head, spatial_mlp_dim)
        self.transformer_enc_temporal = Transformer(temporal_dim, temporal_depth, temporal_heads, temporal_dim_head, temporal_mlp_dim)

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(spatial_dim, temporal_dim),
                nn.Linear(temporal_dim, spatial_dim),
                PreNorm(temporal_dim, CrossAttention(temporal_dim, heads = cross_attn_heads, dim_head = temporal_dim_head, dropout = dropout)),
                nn.Linear(temporal_dim, spatial_dim),
                nn.Linear(spatial_dim, temporal_dim),
                PreNorm(spatial_dim, CrossAttention(spatial_dim, heads = cross_attn_heads, dim_head = spatial_dim_head, dropout = dropout)),
            ]))

    def forward(self, xs, xl):

        xs = self.transformer_enc_spatial(xs)
        xl = self.transformer_enc_temporal(xl)

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            spatial_class = xs[:, 0]
            x_spatial = xs[:, 1:]
            temporal_class = xl[:, 0]
            x_temporal = xl[:, 1:]

            # Cross Attn for Large Patch

            cal_q = f_ls(temporal_class.unsqueeze(1))
            print("cal_q class :", cal_q.shape)
            cal_qkv = torch.cat((cal_q, x_spatial.transpose(0, 1)), dim=1)
            print("cal_qkv class :", cal_qkv.shape)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_temporal), dim=1)


            # Cross Attn for Smaller Patch
            cal_q = f_sl(spatial_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_temporal.transpose(0, 1)), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_spatial), dim=1)

        return xs, xl





class CrossViT(nn.Module):
    def __init__(self, image_size, channels, num_classes, patch_size_spatial = 14, frames = 16, spatial_dim = 192,
                 spatial_depth = 4, temporal_dim=192, temporal_depth = 1, cross_attn_depth = 1, multi_scale_enc_depth = 3,
                 heads = 3, pool = 'cls', dropout = 0., emb_dropout = 0., scale_dim = 4):
        super().__init__()

        assert image_size % patch_size_spatial == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size_spatial) ** 2
        patch_dim_spatial = channels * patch_size_spatial ** 2


        dim = spatial_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b f c (h p1) (w p2) -> b f (h w) (p1 p2 c)', p1 = patch_size_spatial, p2 = patch_size_spatial, f = frames),
            nn.Linear(patch_dim_spatial, dim),
        )


        self.pos_embedding_temporal = nn.Parameter(torch.randn(1, frames + 1, dim))
        self.cls_token_temporal = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_temporal = nn.Dropout(emb_dropout)

        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_spatial = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_spatial = nn.Dropout(emb_dropout)

        self.multi_scale_transformers = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(MultiScaleTransformerEncoder(spatial_dim=spatial_dim, spatial_depth=spatial_depth,
                                                                              spatial_heads=heads, spatial_dim_head=spatial_dim//heads,
                                                                              spatial_mlp_dim=spatial_dim*scale_dim,
                                                                              temporal_dim=temporal_dim, temporal_depth=temporal_depth,
                                                                              temporal_heads=heads, temporal_dim_head=temporal_dim//heads,
                                                                              temporal_mlp_dim=temporal_dim*scale_dim,
                                                                              cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                                                                              dropout=dropout))

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm((num_patches+frames)*spatial_dim),
            nn.Linear((num_patches+frames)*spatial_dim, num_classes)
        )


    def forward(self, img):

        x = self.to_patch_embedding(img)
        b, t, n, _ = x.shape

        xs = rearrange(x, 'b t n d -> (b t) n d')
        bs, ns, _ = xs.shape
        cls_token_spatial = repeat(self.cls_token_spatial, '() n d -> b n d', b = bs)
        xs = torch.cat((cls_token_spatial, xs), dim=1)
        xs += self.pos_embedding_spatial[:, :(n + 1)]
        xs = self.dropout_spatial(xs)

        xt = rearrange(x, 'b t n d -> (b n) t d')
        bt, nt, _ = xt.shape

        cls_token_temporal = repeat(self.cls_token_temporal, '() n d -> b n d', b=bt)
        xt = torch.cat((cls_token_temporal, xt), dim=1)
        xt += self.pos_embedding_temporal[:, :(t + 1)]
        xt = self.dropout_temporal(xt)

        for multi_scale_transformer in self.multi_scale_transformers:
            xs, xt = multi_scale_transformer(xs, xt)
        
        xs = xs.mean(dim = 1) if self.pool == 'mean' else xs[:, 0]
        xt = xt.mean(dim = 1) if self.pool == 'mean' else xt[:, 0]
        xs = rearrange(xs, '(b t) d -> b t d', b=b, t=t)
        xt = rearrange(xt, '(b n) d -> b n d', b=b, n=n)

        x = torch.cat((xs, xt), dim=1)
        x = rearrange(x, 'b n d -> b (n d)')
        x = self.mlp_head(x)

        return x
    
    
    

if __name__ == "__main__":
    
    img = torch.ones([1, 4, 3, 224, 224])
    
    model = CrossViT(224, 3, 1000, 16, 4)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]

    
