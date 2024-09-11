import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from mctf import mctf

class Mlp(nn.Module):
    def __init__(self, inc, hidden=None, outc=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        outc = outc or inc
        hidden = hidden or inc
        self.fc1 = nn.Linear(inc, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, outc)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module): ### one-step ahead 가능하게 변형 필요
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 info = None): ### qk_scale?
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.info = info

    def forward(self, x, size = None, _attn = False):
        B, N, C = x.shape
        if type(_attn) == bool:
            if _attn:
                qk = (x @ self.qkv.weight.T[:, : 2 * C] + self.qkv.bias[:2 * C]).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k = qk[0], qk[1]
            else:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale

            if size is not None:
                attn = attn + size.log()[:, None, None, :, 0]

            attn = attn.softmax(dim=-1)

            attn_ = attn

            if _attn:
                return attn_
            
        else:
            v = (x @ self.qkv.weight.T[:, 2 * C:] + self.qkv.bias[2 * C:]).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            attn = _attn
            attn_ = attn.detach()

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 opt=None,
                 r = 0,
                 notThisTransformer = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(inc=dim, hidden=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.opt = opt
        self.r = r
        if self.opt.use_mctf == True:
            self.use_mctf = True
        else:
            self.use_mctf = False

        if notThisTransformer == True:
            self.use_mctf = False

    def forward(self, x): # x: [B, N, C]
        _, _, C = x.shape
        if C == 512: # out_7 mctf
            size = self.opt.size_7 if self.opt.prop_attn else None
        elif C == 320: # out_14 mctf
            size = self.opt.size_14 if self.opt.prop_attn else None
        elif C == 128: # out_28 mctf
            size = self.opt.size_28 if self.opt.prop_attn else None
        elif C == 64: # out_56 mctf
            size = self.opt.size_56 if self.opt.prop_attn else None
        else:
            size = self.opt.size if self.opt.prop_attn else None

        ###(주석처리) if self.opt["use_mctf"] and self.layer in self.opt["activate"] and self.opt["one_step_ahead"]: ### self.layer in self.opt["activate"] 부분 용도 모름
        if self.use_mctf and self.opt.one_step_ahead:
            if self.opt.one_step_ahead == 1:
                '''### test code
                print("one_step_ahead == 1")
                print("shape of x:", x.shape)
                ###'''
                
                attn = self.attn(self.norm1(x), size, _attn=True) ### attn: [1, 1, N, N]
                x, _attn = mctf(x, attn, self.opt, self.r, self.training) ### x: [1, N - r_eval, 256], _attn: [1, 1, N - r_eval, N - r_eval]
                '''### test code``
                print("shape of attn:", attn.shape)
                print("after mctf")
                print("shape of x:", x.shape)
                print("shape of _attn:", _attn.shape)
                ###'''

                x_attn, _ = self.attn(self.norm1(x), size, _attn = _attn) # x_attn: [1, 176, 256]
                '''###test code
                print("shape of x_attn:", x_attn.shape)
                ###'''

            elif self.opt.one_step_ahead == 2: ### 이해 x
                with torch.no_grad():
                    attn = self.attn(self.norm1(x), size, _attn=True)

                x, _ = mctf(x, attn, self.opt, self.r, self.training)
                size = self.opt.size if self.opt.prop_attn else None
                x_attn, _ = self.attn(self.norm1(x), size)

        else:
            '''### test code
            print("normal SA")
            '''
            x_attn, attn = self.attn(self.norm1(x))

        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        ### if self.opt["use_mctf"] and self.layer in self.opt["activate"] and not self.opt["one_step_ahead"]: # W/O One-Step Ahead
        if self.use_mctf and not self.opt.one_step_ahead: # W/O One-Step Ahead
            x, _ = mctf(x, attn, self.opt, self.r, self.training)
        return x


class Transformer(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, num_patches, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 opt = None,
                 r = 0,
                 notThisTransformer = False):

        super(Transformer, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                opt=opt,
                r = r,
                notThisTransformer = notThisTransformer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim),
                                      requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, peb=True):
        # receive x in shape of B,HW,C
        if peb:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x