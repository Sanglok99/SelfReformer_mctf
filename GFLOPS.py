import torch
import torchprofile
from model.network_test import Net

model = Net()
inputs = torch.randn(1, 3, 224, 224)  # 예시 입력, 모델의 입력 크기에 맞게 변경

flops = torchprofile.profile_macs(model, inputs)  # MACs를 계산
gflops = flops / 1e9  # GFLOPs로 변환
print(f"GFLOPs: {gflops}")