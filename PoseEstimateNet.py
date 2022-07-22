import torch
from torch import nn


"""
构建人体姿态识别的网络结构
   MobileNet V1--------->Cpm--------->initial_stage----------->refine_stage(4-5个）
   输出512维          降维到128      权值共享获得heatmaps和pafs
"""


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )



class Cpm(nn.Module):
    """
    定义Cpm网络，包括conv和conv_dw_no_bn
    由于mobilenet V1输出为512维，有一个cpm的降维层，降维到128维
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    """
    网络包括3部分：
        1).trunk部分：由3个conv组成；
        2).heatmaps部分：由2个conv组成；
        3).pafs部分：由2个conv组成；
    """
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        """
        initial_stage阶段
        :param num_channels: 通道数
        :param num_heatmaps: 热图数
        :param num_pafs: pafs数 = 热图数*2
        """
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    """
    定义RefinementStageBlock网络，是RefinementStage的一部分
    主要包括2部分：
            1). initial部分：conv
            2). trunk部分：2个conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    """
    定义RefineNet网络
    主要包括3部分：
                1). trunk部分：有5个RefinementStageBlock组成；
                2). heatmaps部分：有2个conv组成；
                3). pafs部分：有2个conv组成；

    refine stage包括5个相同的RefinementStageBlock，用于权值共享
    """
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationNet(nn.Module):
    """
    定义姿态评估网络，主要包括4部分：
                    1). mobilenet网络提取特征；
                    2). cpm网络，进一步对特征进行整合
                    3). InitialStage阶段，构建初步的heatmap和pafs
                    4). RefineStage阶段
    """
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        """
        构建人体姿态估计的网络结构
        :param num_refinement_stages: refinement阶段的个数
        :param num_channels: 特征的个数. cpm阶段的输出channel数
        :param num_heatmaps: 热图的个数.
        :param num_pafs: num_pafs的个数. 其为(x,y)最表组合, 因此等于num_heatmaps*2
        """
        # super().__init__()
        super(PoseEstimationNet, self).__init__()
        # 构建mobilenet的前conv5_5
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        # 使用cpm对特征进行进一步的整合
        self.cpm = Cpm(512, num_channels)
        # InitialStage阶段构建初步的heatmap和pafs
        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        # refinestage可以具有多个阶段
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        # backbone
        backbone_features = self.model(x)
        # cpm
        backbone_features = self.cpm(backbone_features)
        # initialstage
        stages_output = self.initial_stage(backbone_features)
        # refine stage
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                # torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)
                # 按照channel将feature, heatmap, pafs进行拼接.
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        # stages_output = stages_output[-2:]

        # one_output = torch.cat(stages_output, dim =1)

        return stages_output


