import os
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LogWritter, calculate_mae
from data import generate_loader
from loss_fn import ConfidentLoss
from tqdm import tqdm


class Solver():
    def __init__(self, module, opt):
        self.opt = opt
        self.logger = LogWritter(opt)
        
        self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")

        print("torch.cuda.is_available: ", torch.cuda.is_available())

        self.net = module.Net(opt) ## 모델 초기화
        self.net = self.net.to(self.dev) ## 디바이스에 로드
            
        msg = "# params:{}\n".format(sum(map(lambda x: x.numel(), self.net.parameters())))
        print(msg)
        self.logger.update_txt(msg)

        self.loss_fn = ConfidentLoss(lmbd=opt.lmbda)
        
        # gather parameters
        base, head = [], []
        for name, param in self.net.named_parameters():
            if "encoder" in name:
                base.append(param)
            else:
                head.append(param)
        assert base!=[], 'encoder is empty'

        self.optim = torch.optim.Adam([{'params':base},{'params':head}], opt.lr,betas=(0.9, 0.999), eps=1e-8) 

        self.train_loader = generate_loader("train", opt) ## 훈련 데이터 로더 설정
        self.eval_loader = generate_loader("test", opt) ## 평가 데이터 로더 설정

        self.best_mae, self.best_step = 1, 0  ## best mae와 best step 초기화

    def fit(self):
        opt = self.opt
        
        for step in range(self.opt.max_epoch):
            #  assign different learning rate
            power = (step+1)//opt.decay_step
            self.optim.param_groups[0]['lr'] = opt.lr * 0.1 * (0.5 ** power)   # for base
            self.optim.param_groups[1]['lr'] = opt.lr * (0.5 ** power)         # for head
            ## 에포크에 따라 학습률 조정(power를 통해)
            print('LR base: {}, LR head: {}'.format(self.optim.param_groups[0]['lr'],
                                                    self.optim.param_groups[1]['lr']))
            print("epoch: {}".format(step))
            for i, inputs in enumerate(tqdm(self.train_loader)): ## enumerate: index, 데이터 반환. tqdm: 진행률 표시줄 보여줌
                self.optim.zero_grad() ## 기울기 누적 방지하기 위해 배치마다 기울기 누적 초기화
                MASK = inputs[0].to(self.dev)
                IMG = inputs[1].to(self.dev)
                ## 데이터를 디바이스로 옮긴다

                pred = self.net(IMG)
                loss, logging = self.loss_fn.get_value(pred, MASK)

                loss.backward()

                if opt.gclip > 0: ## 기울기 클리핑(설정된 값 이상으로 기울기가 증가하지 않도록 함)
                    torch.nn.utils.clip_grad_value_(self.net.parameters(), opt.gclip)

                self.optim.step() ## 옵티마이저 한 스텝 진행

            # eval
            print("[{}/{}]".format(step+1, self.opt.max_epoch))
            self.summary_and_save(step)
            

    def summary_and_save(self, step):
        print('evaluate...')
        mae = self.evaluate()

        if mae < self.best_mae: ## 현재 mae가 이전 epoch의 mae보다 낮다면 체크포인트로 저장
            self.best_mae, self.best_step = mae, step + 1
            self.save(step)
        else: ## 낮지 않아도 save_every_ckpt 옵션이 true 라면 체크포인트로 저장
            if self.opt.save_every_ckpt:
                self.save(step)

        msg = "[{}/{}] MAE: {:.6f} (Best: {:.6f} @ {}K step)\n".format(step+1, self.opt.max_epoch,
                                                                       mae, self.best_mae, self.best_step)
        print(msg)
        self.logger.update_txt(msg)
        ## 현재 에포크 번호, 총 에포크 수, 현재 MAE, 최적의 MAE, 그리고 최적의 MAE를 기록한 에포크의 정보를 msg에 담음
        ## msg를 출력하고 logger에도 기록

    @torch.no_grad()
    def evaluate(self):
        opt = self.opt
        self.net.eval() ## 모델을 평가 모드로 전환

        if opt.save_result: ## 결과 저장할 폴더 생성
            save_root = os.path.join(opt.save_root, opt.dataset)
            os.makedirs(save_root, exist_ok=True)

        mae = 0

        '''
        ### test code
        print_true = False
        ###
        '''

        for i, inputs in enumerate(tqdm(self.eval_loader)):
            MASK = inputs[0].to(self.dev)
            IMG = inputs[1].to(self.dev)
            NAME = inputs[2][0]

            b,c,h,w = MASK.shape

            '''
            ### test code
            if print_true == False:
                print("MASK shape: {}".format(MASK.shape))
                print_true = True
            ###
            '''

            pred = self.net(IMG)

            MASK = MASK.squeeze().detach().cpu().numpy()
            pred_sal = F.pixel_shuffle(pred[-1], 4) # from 56 to 224
            pred_sal = F.interpolate(pred_sal, (h,w), mode='bilinear', align_corners=False)

            pred_sal = torch.sigmoid(pred_sal).squeeze().detach().cpu().numpy()

            if opt.save_result:
                pred_sal = (pred_sal * 255.).astype('uint8')
                save_path_sal = os.path.join(save_root, "{}_sal_eval.png".format(NAME))
                io.imsave(save_path_sal, pred_sal)

            mae += calculate_mae(MASK, pred_sal)
        self.net.train() ## 모델을 다시 학습 모드로 전환

        return mae / len(self.eval_loader) ## 평균 mae 계산하여 return

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        return

    def save(self, step):
        os.makedirs(self.opt.ckpt_root, exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, str(step)+".pt")
        torch.save(self.net.state_dict(), save_path)