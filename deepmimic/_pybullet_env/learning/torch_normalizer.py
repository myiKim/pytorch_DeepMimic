import torch
from _pybullet_env.learning.normalizer import Normalizer
import numpy as np

class TorchNormalizer(Normalizer):

    def __init__(self, size, groups_ids=None, eps=0.02, clip=float('inf')):
        super().__init__(size, groups_ids, eps, clip)

        # self.count = torch.tensor([self.count], dtype=torch.int32)
        # self.mean = torch.tensor(self.mean, dtype=torch.float32)
        # self.std = torch.tensor(self.std, dtype=torch.float32)

        self._build_resource_pt()

    def load(self):
        self.count = self.count_pt.numpy()
        self.mean = self.mean_pt.numpy()
        self.std = self.std_pt.numpy()


    def update(self):
        super().update()
        self._update_resource_pt()

    def set_mean_std(self, mean, std):
        super().set_mean_std(mean, std)
        self._update_resource_pt()

    def normalize_pt(self, x):
        norm_x = (x - self.mean_pt) / self.std_pt
        norm_x = torch.clamp(norm_x, -self.clip, self.clip)
        return norm_x

    def unnormalize_pt(self, norm_x):
        x = norm_x * self.std_pt + self.mean_pt
        return x

    def _build_resource_pt(self):

        # self.count_pt = torch.tensor([self.count], dtype=torch.int32, requires_grad=False).cpu()
        # self.mean_pt = torch.tensor(self.mean, dtype=torch.float32, requires_grad=False).cpu()
        # self.std_pt = torch.tensor(self.std, dtype=torch.float32, requires_grad=False).cpu()
        self.count_pt = torch.tensor([self.count], dtype=torch.int32, requires_grad=False)
        self.mean_pt = torch.tensor(self.mean, dtype=torch.float32, requires_grad=False)
        self.std_pt = torch.tensor(self.std, dtype=torch.float32, requires_grad=False)

    def _update_resource_pt(self):
        # print("Now: ", self.count_pt, self.mean_pt, self.std_pt)
        self.count_pt[0] = self.count
        # print("self.mean : ", self.mean, "&& self.mean_pt: ", self.mean_pt)
        self.mean_pt =  torch.tensor(self.mean, dtype=torch.float32, requires_grad=False)
        # print("self.mean_pt: ", self.mean_pt)
        self.std_pt =  torch.tensor(self.std, dtype=torch.float32, requires_grad=False)
        # vtest = torch.from_numpy(self.mean)
        # print("1", self.mean_pt is self.mean) #False
        # print("2", vtest is self.mean) #False
