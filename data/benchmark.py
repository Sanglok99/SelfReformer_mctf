import os
import glob
import data


# for vairous benchmark datasets
class Benchmark(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.dataset_root
        if phase == "test" and opt.test_dataset != "": ## test phase이고 test_dataset이 빈 값이 아닐 때
            self.name = opt.test_dataset.split('_')[1] ## option.py의 --test_dataset 값(benchmark_DUTSTE)에서 DUTSTE 부분을 name에 넣음
        else:
            self.name = opt.dataset.split('_')[1]

        dir_MASK, dir_IMG = self.get_subdir() ## dir_MASK, dir_IMG 는 각각 benchmark/DUTSTE/Masks 와 benchmark/DUTSTE/Images 로 정해진다
        if self.name == 'HKUIS':
            self.MASK_paths = sorted(glob.glob(os.path.join(root, dir_MASK, "*.png")))
            self.IMG_paths = sorted(glob.glob(os.path.join(root, dir_IMG, "*.png")))
        else: ## 최종적인 평가 데이터셋 경로 결정: .../dataset/benchmark/DUTSTE/...
            self.MASK_paths = sorted(glob.glob(os.path.join(root, dir_MASK, "*.png")))
            self.IMG_paths = sorted(glob.glob(os.path.join(root, dir_IMG, "*.jpg")))

        super().__init__(phase, opt)

    def get_subdir(self):
        dir_MASK = "benchmark/{}/Masks".format(self.name)
        dir_IMG = "benchmark/{}/Images".format(self.name)
        return dir_MASK, dir_IMG