import os
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchprofile
from data import generate_loader
from tqdm import tqdm
from utils import calculate_mae, fbeta_measure
from sklearn.metrics import precision_score, recall_score
from torchvision import transforms
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table


class Tester():
    def __init__(self, module, opt):
        self.opt = opt
        self.cuda = torch.cuda.is_available()  # self.cuda 속성을 추가합니다.

        self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt)
        self.net = self.net.to(self.dev)

        msg = "# params:{}\n".format(
            sum(map(lambda x: x.numel(), self.net.parameters())))
        print(msg)

        self.test_loader = generate_loader("test", opt)

    def _eval_e(self, y_pred, y, num):
        if self.cuda:
            score = torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            score = torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_pred_th = (y_pred >= thlist[i]).float()
            fm = y_pred_th - y_pred_th.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
        return score

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h*w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
        
        aplha = 4 * x * y *sigma_xy
        beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h*w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.from_numpy(np.arange(0,cols)).cuda().float()
                j = torch.from_numpy(np.arange(0,rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0,cols)).float()
                j = torch.from_numpy(np.arange(0,rows)).float()
            X = torch.round((gt.sum(dim=0)*i).sum() / total)
            Y = torch.round((gt.sum(dim=1)*j).sum() / total)
        return X.long(), Y.long()

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
        # print(Q)
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        
        return score

    def _S_object(self, pred, gt):
        fg = torch.where(gt==0, torch.zeros_like(pred), pred)
        bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1-gt)
        u = gt.mean()
        Q = u * o_fg + (1-u) * o_bg
        return Q


    def Eval_Smeasure(self, pred_list, mask_list): # S-measure
        alpha, avg_q, img_num = 0.5, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in zip(pred_list, mask_list):
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                y = gt.mean()
                if y == 0:
                    x = pred.mean()
                    Q = 1.0 - x
                elif y == 1:
                    x = pred.mean()
                    Q = x
                else:
                    gt[gt>=0.5] = 1
                    gt[gt<0.5] = 0
                    #print(self._S_object(pred, gt), self._S_region(pred, gt))
                    Q = alpha * self._S_object(pred, gt) + (1-alpha) * self._S_region(pred, gt)
                    if Q.item() < 0:
                        Q = torch.FloatTensor([0.0])
                img_num += 1.0
                avg_q += Q.item()
            avg_q /= img_num
            return avg_q
        

    def Eval_Emeasure(self, pred_list, mask_list): # E-measure
        avg_e, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            scores = torch.zeros(255)
            if self.cuda:
                scores = scores.cuda()
            for pred, gt in zip(pred_list, mask_list):
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                scores += self._eval_e(pred, gt, 255)
                img_num += 1.0
                
            scores /= img_num
            return scores.max().item()

    @torch.no_grad()
    def evaluate(self, path):
        opt = self.opt

        try:
            print('loading model from: {}'.format(path))
            self.load(path)
        except Exception as e:
            print(e)

        self.net.eval() ## 평가 모드로 전환

        model = self.net
        inputs_Gflop = torch.randn(1,3,224,224).to(self.dev)

        if opt.save_result:
            save_root = os.path.join(opt.save_root, opt.save_msg)
            os.makedirs(save_root, exist_ok=True)

        mae = 0
        
        '''
        ### (MASK 형태 출력)
        ### test code ###
        print_count = 0 # 10번 출력
        ###
        '''

        '''### fbeta_measure ###
        MASK = []
        IMG = []
        NAME = []
        pred_list = []
        mask_list = []
        ### ###'''
    
        for i, inputs in enumerate(tqdm(self.test_loader)):
            MASK = inputs[0].to(self.dev) ### MAE
            IMG = inputs[1].to(self.dev) ### MAE
            NAME = inputs[2][0] ### MAE
            # MASK.append(inputs[0].to(self.dev)) ### fbeta measure
            # IMG.append(inputs[1].to(self.dev)) ### fbeta measure
            # NAME.append(inputs[2][0]) ### fbeta measure

            b, c, h, w = MASK.shape ### MAE
            # b, c, h, w = MASK[i].shape ### fbeta measure
            '''### test code
            print("h, w:", h, w)
            ###'''

            '''
            ### test code (MASK 형태 10개만 출력)
            if print_count < 10:
                print("MASK shape: {} ".format(MASK.shape))
                print_count = print_count + 1
            ###
            '''

            '''
            ### (모든 테스트 데이터에 대해 MASK 형태 출력)
            ### test code
            print("MASK shape:{}".format(MASK.shape))
            ###
            '''
            
            pred = self.net(IMG) ### MAE
            # pred = self.net(IMG[i]) ### fbeta measure
            
            mask = (MASK*255.).squeeze().detach().cpu().numpy().astype('uint8') ### MAE
            # mask = (MASK[i]*255.).squeeze().detach().cpu().numpy().astype('uint8') ### fbeta measure

            ### print("shape of pred[-1]:", pred[-1].shape) ###[1, 16, 56, 56]
            pred_sal = F.pixel_shuffle(pred[-1], 4)
            ### pixel shuffle: [1, 16, 56, 56] -> [1, 1, 56*4, 56*4] -> [1, 1, 224, 224]
            ### print("shape of pred[-1] after pixel shuffle:", pred_sal.shape) ### [1, 1, 224, 224]

            pred_sal = F.interpolate(pred_sal, (h,w), mode='bilinear', align_corners=False)
            ### print("shape of pred[-1] after interpolate:", pred_sal.shape) ### [1, 1, h, w]

            ### print("values of pred_sal before sigmoid:", pred_sal) ### sigmoid 효과 보기(아래 주석의 결과값 참고)
            pred_sal = torch.sigmoid(pred_sal).squeeze()
            ### sigmoid: 출력값이 0부터 1까지
            ### squeeze: 차원이 1인 차원을 제거해준다(주의점: batch가 1일 때 batch 차원도 없어질 수 있음)
            ### print("values of pred_sal after sigmoid:", pred_sal) ### sigmoid 효과 보기(아래 주석의 결과값 참고)
            ### print("shape of pred_sal after sigmoid and squeeze:", pred_sal.shape) ### [h, w] ### squeeze 효과 보기
            '''
            values of pred_sal before sigmoid: tensor([[[[-5.9035, -6.2686, -6.8698,  ..., -7.4717, -6.8519, -6.4755],
                [-6.4666, -6.8992, -7.6117,  ..., -8.2390, -7.5023, -7.0551],
                [-7.1269, -7.6209, -8.4346,  ..., -9.0047, -8.2019, -7.7144],
                ...,
                [-7.4333, -7.8287, -8.4800,  ..., -8.1937, -7.4035, -6.9238],
                [-6.8179, -7.1328, -7.6513,  ..., -7.2408, -6.5704, -6.1633],
                [-6.2584, -6.5067, -6.9156,  ..., -6.4325, -5.8747, -5.5360]]]], device='cuda:0')
            values of pred_sal after sigmoid: tensor([[0.0027, 0.0019, 0.0010,  ..., 0.0006, 0.0011, 0.0015],
                [0.0016, 0.0010, 0.0005,  ..., 0.0003, 0.0006, 0.0009],
                [0.0008, 0.0005, 0.0002,  ..., 0.0001, 0.0003, 0.0004],
                ...,
                [0.0006, 0.0004, 0.0002,  ..., 0.0003, 0.0006, 0.0010],
                [0.0011, 0.0008, 0.0005,  ..., 0.0007, 0.0014, 0.0021],
                [0.0019, 0.0015, 0.0010,  ..., 0.0016, 0.0028, 0.0039]], device='cuda:0')
            '''

            pred_sal = (pred_sal * 255.).detach().cpu().numpy().astype('uint8')
            ### pred_sal * 255: 0과 1로 정규화된 값들을 픽셀 값으로 변환
            ### numpy(): pytorch 텐서를 numpy 배열로 변환
            ### print("shape of pred_sal after '* 255.).detach().cpu().numpy().astype('uint8')':", pred_sal.shape) ### (h, w)

            '''(어디쓰는건지 기억이 안남)
            mask_list.append(mask)
            pred_list.append(pred_sal)
            '''

            '''
            ### fbeta measure ###
            mask_binary = (mask > 127).astype(int)
            pred_binary = (pred_sal > 127).astype(int)
            mask_list.extend(mask_binary.flatten())
            pred_list.extend(pred_binary.flatten())
            ### ###
            '''
            
            matt_img = pred[0].repeat(1,256,1,1)
            ### print("shape of pred[0]:", pred[0].shape) ### [1, 1, 14, 14] ### (pred[0]는 matt)
            ### print("shape of matt_img(after repeat):", matt_img.shape) ### [1, 256, 14, 14]

            matt_img = F.pixel_shuffle(matt_img, 16) ### [1, 256, 14, 14] -> [1, 1, 16*14, 16*14] -> [1, 1, 224, 224]
            ### print("shape of matt_img after pixel_shuffle:", matt_img.shape) ### [1, 1, 224, 224]

            matt_img = F.interpolate(matt_img, (h,w), mode='bilinear', align_corners=False)
            ### print("shape of matt_img after interpolate", matt_img.shape) ### [1, 1, h, w]

            matt_img = torch.sigmoid(matt_img)
            matt_img = (matt_img*255.).squeeze().detach().cpu().numpy().astype('uint8')

            if opt.save_result:
                save_path_msk = os.path.join(save_root, "{}_msk.png".format(NAME)) ### MAE
                save_path_matt = os.path.join(save_root, "{}_matt.png".format(NAME)) ### MAE
                # save_path_msk = os.path.join(save_root, "{}_msk.png".format(NAME[i])) ### fbeta measure
                # save_path_matt = os.path.join(save_root, "{}_matt.png".format(NAME[i])) ### fbeta measure
                io.imsave(save_path_msk, mask)
                # io.imsave(save_path_msk, mask_list[i])
                io.imsave(save_path_matt, matt_img)
                
                if opt.save_all:
                    for idx, sal in enumerate(pred[1:]):
                        scale=224//(sal.shape[-1])
                        sal_img = F.pixel_shuffle(sal,scale)
                        sal_img = F.interpolate(sal_img, (h,w), mode='bilinear', align_corners=False)
                        sal_img = torch.sigmoid(sal_img)
                        sal_path = os.path.join(save_root, "{}_sal_{}.png".format(NAME, idx))
                        sal_img = sal_img.squeeze().detach().cpu().numpy()
                        sal_img = (sal_img * 255).astype('uint8')
                        io.imsave(sal_path, sal_img)
                else:
                    # save pred image
                    save_path_sal = os.path.join(save_root, "{}_sal.png".format(NAME))
                    io.imsave(save_path_sal, pred_sal)

            mae += calculate_mae(mask, pred_sal) ### MAE 계산
            
        # precision = precision_score(mask_list, pred_list)
        # recall = recall_score(mask_list, pred_list)

        flops = torchprofile.profile_macs(model, inputs_Gflop)  # MACs를 계산
        gflops = flops / 1e9  # GFLOPs로 변환
        print(f"GFLOPs: {gflops}")

        return mae/(len(self.test_loader)*255.) ### MAE 반환
        # return fbeta_measure(precision, recall, 0.3)
        # return self.Eval_Emeasure(pred_list, mask_list)

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        return