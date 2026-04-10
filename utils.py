import os
import torch
import shutil
import numpy as np
from torch import nn
from model import (ocean_model)


def set_seed(seed):
    """
    设置随机种子
    Args:
        seed: 随机种子

    Returns: None

    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(output_c: int, input_c: int = 5, dropout_p: float = 0.2):
    """
    output_c: 输出通道数（你的标签通道数，比如 73）
    input_c : 输入通道数（你的输入是 5）
    """
    return ocean_model(in_channels=input_c, out_channels=output_c, dropout_p=dropout_p)


def model_parallel(args, model):
    device_ids = [i for i in range(len(args.gpu.split(',')))]
    model = nn.DataParallel(model, device_ids=device_ids)

    return model


def remove_dir_and_create_dir(dir_name):
    """
    清除原有的文件夹，并且创建对应的文件目录
    Args:
        dir_name: 该文件夹的名字

    Returns: None

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")