import torch
import torch.nn as nn
import math


class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        super().__init__()
        self.emb = in_dim * head_cnt  # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        y = self.dp(self.proj(y))
        return y

    def forward(self, x):
        x = x + self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Fuser(nn.Module):
    def __init__(self, emb_dim=320, hw=7, cur_stg=512):
        super(Fuser, self).__init__()

        self.shuffle = nn.PixelShuffle(2)
        self.unfold = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.concatFuse = nn.Sequential(nn.Linear(emb_dim + cur_stg // 4, emb_dim),
                                        nn.GELU(),
                                        nn.Linear(emb_dim, emb_dim))
        self.att = Token_performer(dim=emb_dim, in_dim=emb_dim, kernel_ratio=0.5)
        self.hw = hw

    def forward(self, a, b):
        ### -> test code
        
        B, _, _ = b.shape ### a: [1, 196, 320], b: [1, 49, 512]
        b = self.shuffle(b.transpose(1, 2).reshape(B, -1, self.hw, self.hw))
        ### print("shape of b after shuffle:", b.shape) ### [1, 128, 14, 14]
            ### [1, 49, 512] ->(transpose)-> [1, 512, 49] ->(reshape)-> [1, 512, 7, 7] ->(shuffle)-> [1, 128, 14, 14]
        b = self.unfold(b).transpose(1, 2)
        ### print("shape of b after unfold:", b.shape) ### [1, 196, 128]
            ### [1, 128, 14, 14] ->(unfold)-> [1, 128, 196] ->(transpose)-> [1, 196, 128]

        out = self.concatFuse(torch.cat([a, b], dim=2))
        ### print("shape of out after concatFuse:", out.shape) ### [1, 196, 320]
            ### a: [1, 196, 320], b: [1, 196, 128] ->(cat)-> [1, 196, 448] ->(concatFuse)-> [1, 196, 320]
            ### concatFuse: Linear(320(em_dim) + 128(512(cur_stg)//4), emb_dim) 이라 448 -> 320
        out = self.att(out) ### self attention? 까지 하고 나오는듯
        ### print("shape of out after att:", out.shape) ### [1, 196, 320] (똑같음)

        return out
    

class mctf_token_recover(nn.Module):
    def __init__(self, r_eval = 0, num_patches = 196):
        super(mctf_token_recover, self).__init__()

        self.r = 2 * r_eval
        self.num_patches = num_patches
        self.token_recover = nn.Sequential(nn.Linear(num_patches - self.r, num_patches),
                                           nn.GELU(),
                                           nn.Linear(num_patches, num_patches))

    def forward(self, x):
        # x in shape of B, N, C
        # ([1, 196 - 2*r_eval, 256])
        _, N, _ = x.shape
        if N == self.num_patches - self.r:
            x = x.transpose(1, 2) # [1, 256, 196 - 2*r_eval]
            x = self.token_recover(x) # [1, 256, 196]
            '''### test code
            print("token recovered(%d - 2*r_eval -> %d)" % (self.num_patches, self.num_patches))
            ###'''
            x = x.transpose(1, 2) # [1, 196, 256]

            return x
        else:
            raise ValueError("Expected input with second dimension N equal to (%d - 2*r_eval)." % self.num_patches)