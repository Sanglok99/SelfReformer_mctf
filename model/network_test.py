from .Encoder_pvt import Encoder
from .Transformer import Transformer
from .CRM import CRM
from .Fuser import Fuser, mctf_token_recover
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.opt = opt

        self.encoder = Encoder()
        # global context
        self.encoder_tf_ss = Transformer(depth=2,
                                         num_heads=1,
                                         embed_dim=256,
                                         mlp_ratio=3,
                                         num_patches=196,
                                         opt=opt,
                                         notThisTransformer=True
                                         )

        self.encoder_shaper_7 = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1024), nn.GELU())
        self.encoder_shaper_14 = nn.Sequential(nn.LayerNorm(320), nn.Linear(320, 256), nn.GELU())
        self.encoder_shaper_28 = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU())
        self.encoder_shaper_56 = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 16), nn.GELU())

        self.encoder_merge7_14 = nn.Sequential(nn.BatchNorm2d(512),
                                               nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
                                               nn.LeakyReLU())
        self.encoder_merge28_14 = nn.Sequential(nn.BatchNorm2d(512),
                                                nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
                                                nn.LeakyReLU())
        self.encoder_merge56_14 = nn.Sequential(nn.BatchNorm2d(512),
                                                nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
                                                nn.LeakyReLU())

        self.encoder_pred = nn.Sequential(nn.LayerNorm(256),
                                          nn.Linear(256, 256),
                                          nn.GELU(),
                                          nn.LayerNorm(256),
                                          nn.Linear(256, 256),
                                          nn.GELU(),
                                          nn.LayerNorm(256),
                                          nn.Linear(256, 1)
                                          )
        # main network
        self.transformer_out_7_1 = Transformer(depth=1, num_heads=1, embed_dim=512, mlp_ratio=3, num_patches=49, opt=opt, r=6)
        self.transformer_out_7_2 = Transformer(depth=1, num_heads=1, embed_dim=512, mlp_ratio=3, num_patches=43, opt=opt, r=7)

        self.transformer = nn.ModuleList([Transformer(depth=d,
                                                      num_heads=n,
                                                      embed_dim=e,
                                                      mlp_ratio=m,
                                                      num_patches=p,
                                                      r=r,
                                                      opt=opt) for d, n, e, m, p, r in opt.transformer])

        self.fuser7_14 = Fuser(emb_dim=320, hw=6, cur_stg=512)
        self.fuser14_28 = Fuser(emb_dim=128, hw=12, cur_stg=320)
        self.fuser28_56 = Fuser(emb_dim=64, hw=24, cur_stg=128)

        self.CRM_7 = CRM(inc=512, outc=1024, hw=6, embed_dim=512, num_patches=36, opt=opt)
        self.CRM_14 = CRM(inc=320, outc=256, hw=12, embed_dim=320, num_patches=144, opt=opt)
        self.CRM_28 = CRM(inc=128, outc=64, hw=24, embed_dim=128, num_patches=576, opt=opt)
        self.CRM_56 = CRM(inc=64, outc=16, hw=48, embed_dim=64, num_patches=2304, opt=opt)

    def forward(self, x):
        B = x.shape[0]

        ### print("shape of x:", x.shape)

        
        '''##### (encoder에서 출력 얼마나 걸리는지 측정)
        ##### test code #####
        encoder_start_event = torch.cuda.Event(enable_timing=True)
        encoder_end_event = torch.cuda.Event(enable_timing=True)
        
        encoder_start_event.record()
        ######'''
        

        # PVT encoder
        out_7r, out_14r, out_28r, out_56r = self.encoder(x)  # is a cat_feature, list in shape of 16, 32, 64, 128
        '''###test code
        print("shape of out_7r:", out_7r.shape) # [1, 49, 512]
        print("shape of out_14r:", out_14r.shape) # [1, 196, 320]
        print("shape of out_28r:", out_28r.shape) # [1, 784, 128]
        print("shape of out_56r:", out_56r.shape) # [1, 3136, 64]'''
        
        pred = list()

        
        '''##### test code #####
        encoder_end_event.record()
        torch.cuda.synchronize()
        encoder_time_taken = encoder_start_event.elapsed_time(encoder_end_event)
        print(f"Inference time for the encoder: {encoder_time_taken * 1e-3} seconds") # milliseconds to seconds
        #####


        ##### (global context에서 출력 얼마나 걸리는지 측정)
        ##### test code #####
        global_start_event = torch.cuda.Event(enable_timing=True)
        global_end_event = torch.cuda.Event(enable_timing=True)
        
        global_start_event.record()
        #####'''
        

        # ----------------------------for global context
        # reshape
        
        ### -> test code
        out_7s = self.encoder_shaper_7(out_7r).transpose(1, 2).reshape(B, 1024, 7, 7)
        ### print("shape of out_7s:", out_7s.shape) ### [1, 1024, 7, 7]
        ### [1, 49, 512] ->(encoder_shaper_7)-> [1, 49, 1024] ->(transpose)-> [1, 1024, 49] ->(reshape)-> [1, 1024, 7, 7]
        out_7s = F.pixel_shuffle(out_7s, 2)
        ### print("shape of out_7s after pixel_shuffle:", out_7s.shape) ### [1, 256, 14, 14]

        out_14s = self.encoder_shaper_14(out_14r).transpose(1, 2).reshape(B, 256, 14, 14)
        ### print("shape of out_14s:", out_14s.shape) ### [1, 256, 14, 14]

        out_28s = self.encoder_shaper_28(out_28r).transpose(1, 2).reshape(B, 64, 28, 28)
        ### print("shape of out_28s:", out_28s.shape) ### [1, 64, 28, 28]
        out_28s = F.pixel_unshuffle(out_28s, 2)
        ### print("shape of out_28s after pixel_shuffle:", out_28s.shape) ### [1, 256, 14, 14]

        out_56s = self.encoder_shaper_56(out_56r).transpose(1, 2).reshape(B, 16, 56, 56)
        ### print("shape of out_56s:", out_7s.shape) ### [1, 256, 14, 14]
        out_56s = F.pixel_unshuffle(out_56s, 4)
        ### print("shape of out_56s after pixel_shuffle:", out_56s.shape) ### [1, 256, 14, 14]


        # merge

        ### -> test code
        out = self.encoder_merge7_14(torch.cat([out_14s, out_7s], dim=1))
        out = self.encoder_merge28_14(torch.cat([out, out_28s], dim=1))
        out = self.encoder_merge56_14(torch.cat([out, out_56s], dim=1))
        ### print("shape of out:", out.shape) ### [1, 256, 14, 14]
        out = out.reshape(B, 256, -1).transpose(1, 2)  # B,N,C
        ### print("shape of out after:", out.shape) ### [1, 196, 256]


        # pred

        ### -> test code
        out = self.encoder_tf_ss(out)
        ### print("shape of out after encoder_tf_ss:", out.shape) ### [1, 168, 256](r_eval 14일 때), [1, 156, 256](r_eval 20일 때), [1, 196, 256](MCTF 안썼을때)

        matt = self.encoder_pred(out).transpose(1, 2).reshape(B, 1, 14, 14) ### (token fusion 없이) [1, 196, 1] -> [1, 1, 196] -> [1, 1, 14, 14]
        pred.append(matt)

        '''### test code(evalutation 할 때)
        print("shape of matt:", matt.shape) ### [1, 1, 14, 14]
        ###'''

        ### print("shape of matt before 'REPEAT':", matt.shape) ### [1, 1, 14, 14]
        matt = matt.repeat(1, 256, 1, 1) ### (mctf 하면) [B, 1, 12, 12] -> [B, 256, 12, 12]
        ### print("shape of matt after 'REPEAT':", matt.shape) ### [1, 256, 14, 14]

        ##### matt 7사이즈에서 6사이즈로 바꿔주는 Linear
        matt_1 = F.interpolate(matt, size=(12, 12), mode='bilinear', align_corners=False)
        ### print("shape of matt_1(after interpolate to (12,12)):", matt_1.shape) ### [16, 256, 12, 12]
        matt_1 = F.pixel_shuffle(matt_1, 16)
        ### print("shape of matt_1 after pixel shuffle:", matt_1.shape) ### [16, 1, 192, 192]
        #####

        matt = F.pixel_shuffle(matt, 16)  # B, 1, 192, 192
        ### print("shape of matt after pixel shuffle:", matt.shape) ### [1, 1, 192, 192]

        

        
        '''##### test code #####
        global_end_event.record()
        torch.cuda.synchronize()
        global_time_taken = global_start_event.elapsed_time(global_end_event)
        print(f"Inference time for global context: {global_time_taken * 1e-3} seconds") # milliseconds to seconds
        #####'''


        '''##### (SOD에서 출력 얼마나 걸리는지 측정)
        ##### test code #####
        SOD_start_event = torch.cuda.Event(enable_timing=True)
        SOD_end_event = torch.cuda.Event(enable_timing=True)
        
        SOD_start_event.record()
        #####

        ####### (tf에서 시간 얼마나 걸리는지)
        out_7_to_56_start_event = torch.cuda.Event(enable_timing=True)
        out_7_to_56_end_event = torch.cuda.Event(enable_timing=True)
        out_7_to_56_start_event.record()
        #######'''
        

        # ----------------------------for SOD
        out_7 = self.transformer_out_7_1(out_7r)
        out_7 = self.transformer_out_7_2(out_7)
        out_14, out_28, out_56 = [tf(o, peb) for tf, o, peb in zip(self.transformer, ### nn.ModuleList <- [2, 1, 320, 3, 196], [2, 1, 128, 3, 784], [2, 1, 64, 3, 3136] ### 모두 depth(첫번째 값)는 2, num_heads(두번째 값)는 1, mlp_ratio(네번째 값)는 3
                                                                    [out_14r, out_28r, out_56r], ### [1, 196, 320], [1, 784, 128], [1, 3136, 64]
                                                                    [False, False, False])]  # B, patch, feature
        
        '''#######
        out_7_to_56_end_event.record()
        torch.cuda.synchronize()
        out_7_to_56_time_taken = out_7_to_56_start_event.elapsed_time(out_7_to_56_end_event)
        print(f"Inference time for out_7_to_56: {out_7_to_56_time_taken * 1e-3} seconds") # milliseconds to seconds
        #######'''

        ### 위의 tf에 mctf 적용한다면
        ### out_7 -> [1, 49 - 13, 512]
        ### out_14 -> [1, 196 - 52, 320]
        ### out_28 -> [1, 784 - 208, 128]
        ### out_56 -> [1, 3136 - 832, 64]
        
        '''### test code
        print("shape of out_7:", out_7.shape) ### [1, 49, 512]
        print("shape of out_14:", out_14.shape) ### [1, 196, 320]
        print("shape of out_28:", out_28.shape) ### [1, 784, 128]
        print("shape of out_56:", out_56.shape) ### [1, 3136, 64]
        ###
        '''

        # 7
        p1_7, p2_7, out_7 = self.CRM_7(out_7, matt_1)
        pred.append(p1_7)
        pred.append(p2_7)
        '''### test code
        print("shape of p1_7:", p1_7.shape) ### [1, 1024, 7, 7]
        print("shape of p2_7:", p2_7.shape) ### [1, 1024, 7, 7]
        print("shape of out_7:", out_7.shape) ### [1, 49, 512]'''

        # 14
        out_14 = self.fuser7_14(out_14, out_7)
        ### out_14: [1, 196, 320], out_7: [1, 49, 512], self.fuser7_14 = Fuser(emb_dim=320, hw=7, cur_stg=512)
        ### print("shape of out_14 after fuser:", out_14.shape) ### [1, 196, 320]
        p1_14, p2_14, out_14 = self.CRM_14(out_14, matt_1)
        pred.append(p1_14)
        pred.append(p2_14)
        '''### test code
        print("shape of p1_14:", p1_14.shape) ### [1, 256, 14, 14]
        print("shape of p2_14:", p2_14.shape) ### [1, 256, 14, 14]
        print("shape of out_14:", out_14.shape) ### [1, 196, 320]'''

        # 28
        out_28 = self.fuser14_28(out_28, out_14)
        ### print("shape of out_28 after fuser", out_28.shape) ### [1, 784, 128]
        p1_28, p2_28, out_28 = self.CRM_28(out_28, matt_1)
        pred.append(p1_28)
        pred.append(p2_28)
        '''### test code
        print("shape of p1_28:", p1_28.shape) ### [1, 64, 28, 28]
        print("shape of p2_28:", p2_28.shape) ### [1, 64, 28, 28]
        print("shape of out_28:", out_28.shape) ### [1, 784, 128]'''

        # 56
        out_56 = self.fuser28_56(out_56, out_28)
        ### print("shape of out_56 after fuser", out_56.shape) ### [1, 3136, 64]
        p1_56, p2_56, out_56 = self.CRM_56(out_56, matt_1)
        pred.append(p1_56)
        pred.append(p2_56)
        '''### test code
        print("shape of p1_56:", p1_56.shape) ### [1, 16, 56, 56]
        print("shape of p2_56:", p2_56.shape) ### [1, 16, 56, 56]
        print("shape of out_56:", out_56.shape) ### [1, 3136, 64]'''

        
        '''##### test code #####
        SOD_end_event.record()
        torch.cuda.synchronize()
        SOD_time_taken = SOD_start_event.elapsed_time(SOD_end_event)
        print(f"Inference time for SOD: {SOD_time_taken * 1e-3} seconds") # milliseconds to seconds
        #####'''
        

        ##### mctf 적용
        self.opt.size_7 = None
        self.opt.size_14 = None
        self.opt.size_28 = None
        self.opt.size_56 = None
        self.opt.size = None
        ##### (mctf 적용 후 줄어든 size 값 초기화)

        return pred