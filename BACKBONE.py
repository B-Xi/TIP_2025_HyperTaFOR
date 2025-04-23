import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F


class TopkRouting(nn.Module):
    def __init__(self, qk_dim, topk=4, qk_scale=None):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5

    def forward(self, query, key):
        query_hat, key_hat = query.detach(), key.detach()
        attn_logit = (query_hat * self.scale) @ key_hat.permute([0, 2, 1])  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        return topk_index


class KVGather(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r_idx, kv):
        b, p2, w2, c_kv = kv.shape
        topk = r_idx.shape[-1]
        topk_kv = torch.gather(
            kv.reshape((b, 1, p2, w2, c_kv)).expand((-1, p2, -1, -1, -1)),  # (n, p^2, p^2, w^2, c_kv) without mem cpy
            dim=2,
            index=r_idx.reshape((b, p2, topk, 1, 1)).expand((-1, -1, -1, w2, c_kv)),  # (n, p^2, k, w^2, c_kv)
        )
        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = torch.split(self.qkv(x), [self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=13, qk_dim=None, qk_scale=None, topk=10, side_dwconv=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_win = n_win
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.qk_scale = qk_scale or self.qk_dim ** -0.5
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2, groups=dim)
        self.topk = topk
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.qk_scale,
                                  topk=self.topk)
        self.kv_gather = KVGather()
        self.qkv = QKVLinear(self.dim, self.qk_dim)
        self.wo = nn.Linear(dim, dim)
        self.attn_act = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        n = x.shape[1]
        h = int(pow(n, 0.5))
        x = rearrange(x, "b (h w) c -> b h w c", h=h)  # 65*13*13*64
        N, H, W, C = x.shape
        x = rearrange(x, "b (j h) (i w) c -> b (j i) h w c", j=self.n_win, i=self.n_win)  # 64*169*1*1*64
        q, kv = self.qkv(x)
        q_pix = rearrange(q, 'b p2 h w c -> b p2 (h w) c')  # 65*169*1*64
        kv_pix = rearrange(kv, 'b p2 h w c -> b p2 (h w) c')  # 65*169*1*128
        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])
        lepe = self.lepe(
            rearrange(kv[..., self.qk_dim:], 'b (j i) h w c -> b c (j h) (i w)', j=self.n_win,
                      i=self.n_win))  # 65*64*13*13
        lepe = rearrange(lepe, 'b c h w -> b h w c')  # 65*13*13*64
        r_idx = self.router(q_win, k_win)  # 65*169*10
        kv_pix_sel = self.kv_gather(r_idx=r_idx, kv=kv_pix)  # 65*169*10*1*128  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = torch.split(kv_pix_sel, [self.qk_dim, self.dim], dim=-1)
        k_pix_sel = rearrange(k_pix_sel, 'b p2 k w2 (mh c) -> (b p2) mh c (k w2)', mh=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'b p2 k w2 (mh c) -> (b p2) mh (k w2) c', mh=self.num_heads)
        q_pix = rearrange(q_pix, 'b p2 w2 (mh c) -> (b p2) mh w2 c', mh=self.num_heads)
        attn = q_pix @ k_pix_sel * self.qk_scale
        attn = self.attn_act(attn)
        out = attn @ v_pix_sel
        out = rearrange(out, '(b j i) mh (h w) c -> b (j h) (i w) (mh c)',
                        j=self.n_win, i=self.n_win, h=H // self.n_win, w=W // self.n_win)
        out = out + lepe
        out = self.wo(out)
        out = rearrange(out, "b h w c -> b (h w) c", h=h)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Biformer(nn.Module):
    def __init__(self, dim, depth, heads, patch_size, mlp_dim, dropout,topk=24):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(
                    LayerNormalize(dim, BiLevelRoutingAttention(dim=dim, num_heads=heads, n_win=patch_size, qk_dim=None,
                                                                qk_scale=None, topk=topk, side_dwconv=7))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        x_center = []
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
            index = int(x.shape[1] // 2)
            x_center.append(x[:, index, :])
        return x, x_center


class Backbone(nn.Module):
    def __init__(self, params):
        super(Backbone, self).__init__()
        self.params = params
        patch_size = params.get("patch", 13)
        self.spectral_size = params.get("spectral_size", 100)
        depth = params.get("depth", 1)
        heads = params.get("heads", 8)
        mlp_dim = params.get("mlp_head_dim", 8)
        kernal = params.get('kernal', 3)
        padding = params.get('padding', 1)
        dropout = params.get("dropout", 0)
        conv2d_out = 64
        dim = params.get("dim", 64)
        topk = params.get("topk", 24)
        image_size = patch_size * patch_size
        self.pixel_patch_embedding = nn.Linear(conv2d_out, dim)
        self.local_trans_pixel = Biformer(dim=dim, depth=depth, heads=heads, patch_size=patch_size, mlp_dim=mlp_dim,
                                          dropout=dropout,topk=topk)
        self.new_image_size = image_size
        self.pixel_pos_embedding_relative = nn.Parameter(torch.randn(self.new_image_size, dim))
        self.pixel_pos_scale = nn.Parameter(torch.ones(1) * 0.01)
        self.center_weight = nn.Parameter(torch.ones(depth, 1, 1) * 0.001)
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.spectral_size, out_channels=conv2d_out, kernel_size=(kernal, kernal),
                      padding=(padding, padding)),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.1)

    def get_position_embedding(self, x, center_index, cls_token=False):
        center_h, center_w = center_index
        _, _, h, w = x.shape
        pos_index = []
        for i in range(h):
            temp_index = []
            for j in range(w):
                temp_index.append(max(abs(i - center_h), abs(j - center_w)))
            pos_index.append(temp_index[:])
        pos_index = np.asarray(pos_index)
        pos_index = pos_index.reshape(-1)
        if cls_token:
            pos_index = np.asarray([-1] + list(pos_index))
        pos_emb = self.pixel_pos_embedding_relative[pos_index, :]
        return pos_emb

    def forward(self, x):
        x_pixel = x
        b, s, w, h = x_pixel.shape
        x_pixel = self.conv2d_features(x_pixel)
        pos_emb = self.get_position_embedding(x_pixel, (h // 2, w // 2), cls_token=False)
        x_pixel = rearrange(x_pixel, 'b s w h-> b (w h) s')  # (batch, w*h, s)
        x_pixel = x_pixel + torch.unsqueeze(pos_emb, 0)[:, :, :] * self.pixel_pos_scale
        x_pixel = self.dropout(x_pixel)
        x_pixel, x_center_list = self.local_trans_pixel(x_pixel)  # (batch, image_size+1, dim)
        x_center_tensor = torch.stack(x_center_list, dim=0)  # [depth, batch, dim]
        logit_pixel = torch.sum(x_center_tensor * self.center_weight, dim=0)
        logit_x = logit_pixel
        return logit_x
