import torch
import torch.nn as nn
from .Transformer import Transformer
import torch.nn.functional as F


class CRM(nn.Module):
    def __init__(self, inc, outc, hw, embed_dim, num_patches, depth=4, opt=None):
        super(CRM, self).__init__()
        self.conv_p1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_p2 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_glb = nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True)

        self.conv_matt = nn.Sequential(nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))
        self.conv_fuse = nn.Sequential(nn.Conv2d(2 * inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()
        self.tf = Transformer(depth=depth,
                              num_heads=1,
                              embed_dim=embed_dim,
                              mlp_ratio=3,
                              num_patches=num_patches,
                              opt=opt,
                              notThisTransformer=True)
        self.hw = hw
        self.inc = inc

    def forward(self, x, glbmap): ### glpmap은 matt => 얘는 바꿔줘야할듯
        # x in shape of B,N,C
        # glbmap in shape of B,1,224,224
        B, _, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.hw, self.hw) ## x: [B, N, C] -> [B, C, N] -> [B, C, hw, hw]
        if glbmap.shape[-1] // self.hw != 1: ## glbmap의 hw가 self.hw보다 2배 이상 크면 ### (//: 몫)
            ### print("shape of glpmap before pixel unshuffle:", glbmap.shape)
            glbmap = F.pixel_unshuffle(glbmap, glbmap.shape[-1] // self.hw)
            ### print("shape of glpmap after pixel unshuffle:", glbmap.shape)
            glbmap = self.conv_glb(glbmap)
            ### print("shape of glpmap after conv_glb:", glbmap.shape)
            '''
            shape of glpmap before pixel unshuffle: torch.Size([1, 1, 224, 224])
            shape of glpmap after pixel unshuffle: torch.Size([1, 1024, 7, 7])
            shape of glpmap after conv_glb: torch.Size([1, 512, 7, 7])

            shape of glpmap before pixel unshuffle: torch.Size([1, 1, 224, 224])
            shape of glpmap after pixel unshuffle: torch.Size([1, 256, 14, 14])
            shape of glpmap after conv_glb: torch.Size([1, 320, 14, 14])

            shape of glpmap before pixel unshuffle: torch.Size([1, 1, 224, 224])
            shape of glpmap after pixel unshuffle: torch.Size([1, 64, 28, 28])
            shape of glpmap after conv_glb: torch.Size([1, 128, 28, 28])

            shape of glpmap before pixel unshuffle: torch.Size([1, 1, 224, 224])
            shape of glpmap after pixel unshuffle: torch.Size([1, 16, 56, 56])
            shape of glpmap after conv_glb: torch.Size([1, 64, 56, 56])
            '''

        ### print("shape of glpmap before cat:", glbmap.shape) ###(mctf 적용시) [16, 512, 6, 6], [16, 320, 12, 12], ...
        ### print("shape of x before cat:", x.shape) ###(mctf 적용시) [16, 512, 6, 6], [16, 320, 12, 12], ...
        x = torch.cat([glbmap, x], dim=1)
        ### print("shape of x after cat:", x.shape) ###(mctf 적용시) [16, 1024, 6, 6], [16, 640, 12, 12], ...
        '''
        shape of glpmap before cat: torch.Size([1, 512, 7, 7])
        shape of x before cat: torch.Size([1, 512, 7, 7])
        shape of x after cat: torch.Size([1, 1024, 7, 7])

        shape of glpmap before cat: torch.Size([1, 320, 14, 14])
        shape of x before cat: torch.Size([1, 320, 14, 14])
        shape of x after cat: torch.Size([1, 640, 14, 14])
        
        shape of glpmap before cat: torch.Size([1, 128, 28, 28])
        shape of x before cat: torch.Size([1, 128, 28, 28])
        shape of x after cat: torch.Size([1, 256, 28, 28])

        shape of glpmap before cat: torch.Size([1, 64, 56, 56])
        shape of x before cat: torch.Size([1, 64, 56, 56])
        shape of x after cat: torch.Size([1, 128, 56, 56])
        '''

        '''####### (conv_fuse에서 시간 얼마나 걸리는지)
        conv_fuse_start_event = torch.cuda.Event(enable_timing=True)
        conv_fuse_end_event = torch.cuda.Event(enable_timing=True)
        conv_fuse_start_event.record()
        #######'''

        x = self.conv_fuse(x)

        '''#######
        conv_fuse_end_event.record()
        torch.cuda.synchronize()
        conv_fuse_time_taken = conv_fuse_start_event.elapsed_time(conv_fuse_end_event)
        print(f"Inference time for conv_fuse: {conv_fuse_time_taken * 1e-3} seconds") # milliseconds to seconds
        #######'''

        ### print("shape of x after conv_fuse:", x.shape)
        '''
        shape of x after conv_fuse: torch.Size([1, 512, 7, 7])
        shape of x after conv_fuse: torch.Size([1, 320, 14, 14])
        shape of x after conv_fuse: torch.Size([1, 128, 28, 28])
        shape of x after conv_fuse: torch.Size([1, 64, 56, 56])
        '''

        # pred
        p1 = self.conv_p1(x)
        ### print("shape of p1:", p1.shape)
        '''
        shape of p1: torch.Size([1, 1024, 7, 7])
        shape of p1: torch.Size([1, 256, 14, 14])
        shape of p1: torch.Size([1, 64, 28, 28])
        shape of p1: torch.Size([1, 16, 56, 56])
        '''
        
        matt = self.sigmoid(p1)
        ### print("shape of matt after sigmoid:", matt.shape) ### p1과 동일

        ##### 여기에 p1을 6사이즈에서 7사이즈로 변환해주는 interpolate?
        if C == 512:
            p1 = F.interpolate(p1, size=(7,7), mode = 'bilinear', align_corners=False)
        elif C == 320:
            p1 = F.interpolate(p1, size=(14, 14), mode = 'bilinear', align_corners=False)
        elif C == 128:
            p1 = F.interpolate(p1, size=(28, 28), mode = 'bilinear', align_corners=False)
        elif C == 64:
            p1 = F.interpolate(p1, size=(56, 56), mode = 'bilinear', align_corners=False)
        else:
            raise ValueError
        #####

        matt = matt * (1 - matt)
        ### print("shape of matt after 'H':", matt.shape) ### p1과 동일

        matt = self.conv_matt(matt)
        ### print("shape of matt after conv_matt:", matt.shape)
        '''
        shape of matt after conv_matt: torch.Size([1, 512, 7, 7])
        shape of matt after conv_matt: torch.Size([1, 320, 14, 14])
        shape of matt after conv_matt: torch.Size([1, 128, 28, 28])
        shape of matt after conv_matt: torch.Size([1, 64, 56, 56])
        '''

        fea = x * (1 + matt)
        ### print("shape of fea:", fea.shape) ### 위의 matt과 동일

        # reshape x back to B,N,C
        fea = fea.reshape(B, self.inc, -1).transpose(1, 2)
            ### print("shape of fea after reshape and transpose:", fea.shape)
        '''shape of fea after reshape and transpose: torch.Size([1, 49, 512])
            shape of fea after reshape and transpose: torch.Size([1, 196, 320])
            shape of fea after reshape and transpose: torch.Size([1, 784, 128])
            shape of fea after reshape and transpose: torch.Size([1, 3136, 64])'''
        
        fea = self.tf(fea)
            ### print("shape of fea after tf:", fea.shape) ### 위의 fea 와 동일
        '''shape of fea after tf: torch.Size([1, 49, 512])
            shape of fea after tf: torch.Size([1, 196, 320])
            shape of fea after tf: torch.Size([1, 784, 128])
            shape of fea after tf: torch.Size([1, 3136, 64])'''
        
        p2 = self.conv_p2(fea.transpose(1, 2).reshape(B, C, self.hw, self.hw))
            #### print("shape of p2:", p2.shape)
        '''shape of p2: torch.Size([1, 1024, 7, 7])
            shape of p2: torch.Size([1, 256, 14, 14])
            shape of p2: torch.Size([1, 64, 28, 28])
            shape of p2: torch.Size([1, 16, 56, 56])'''
        

        ##### 여기에 p2를 6사이즈에서 7사이즈로 변환해주는 interpolate?
        if C == 512:
            p2 = F.interpolate(p2, size=(7,7), mode = 'bilinear', align_corners=False)
        elif C == 320:
            p2 = F.interpolate(p2, size=(14, 14), mode = 'bilinear', align_corners=False)
        elif C == 128:
            p2 = F.interpolate(p2, size=(28, 28), mode = 'bilinear', align_corners=False)
        elif C == 64:
            p2 = F.interpolate(p2, size=(56, 56), mode = 'bilinear', align_corners=False)
        else:
            raise ValueError
        #####

        return [p1, p2, fea]
