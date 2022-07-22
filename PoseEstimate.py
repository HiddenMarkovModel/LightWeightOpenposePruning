#coding:utf8
from operator import itemgetter
from pathlib import Path
import cv2
import math
import numpy as np
import torch
from PoseEstimateNet import PoseEstimationNet


class PoseEstimate:

    def __init__(self, model_name="checkpoint_iter_370000.pth", device='cpu', height=256):
        """

        :param model_name: 预训练模型名称. 位于checkpoints中.
        :param device: 推理的设备: cpu或cuda
        """
        self.model_name = Path(__file__).parent.joinpath("checkpoints", model_name)
        self.device = torch.device("cuda") if device=='cpu' and torch.cuda.is_available() else torch.device('cpu')
        self.model = self.__buildModel()
        self.height = height

    def getModel(self):

        return self.model

        # from torchvision.models import mobilenet_v2
        # self.model = mobilenet_v2(pretrained=True)
        #
        # return self.model.model

        # trunk = self.model.cpm.trunk[0][:1]
        #
        # return trunk


    def __buildModel(self):
        """
        构建模型
        :return:
        """
        model = PoseEstimationNet(num_refinement_stages=3)
        model.load_state_dict(torch.load(str(self.model_name), map_location='cpu')['state_dict'])

        model.to(self.device)

        return model

