import torch
import time
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.pruning import L1NormPruner
from pathlib import Path

from PoseEstimate import PoseEstimate


if __name__ == '__main__':


    model = PoseEstimate().getModel()

    print("*******************原始模型*************************")
    print(model)

    # torch.save(model.state_dict(), str(save_root.joinpath("openpose_orig.pth")))

    config_list = [
        {"total_sparsity": 0.1,
         'op_types': ['Conv2d']
         },
    ]

    pruner = L1NormPruner(model, config_list)
    _, mask = pruner.compress()
    pruner._unwrap_model()

    dummy_input = torch.rand(1, 3, 256, 256)
    ModelSpeedup(model, dummy_input, mask).speedup_model()

    print("*******************加速完成*************************")
    print(model)
