import os
import random
import numpy as np
import torch


def set_random_seed(seed=520):
    """设置所有随机种子以确保实验可重复性"""
    # Python随机种子
    random.seed(seed)

    # NumPy随机种子
    np.random.seed(seed)

    # PyTorch随机种子
    torch.manual_seed(seed)

    # 如果是CUDA环境
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        torch.backends.cudnn.deterministic = True  # 使用确定性卷积算法
        torch.backends.cudnn.benchmark = False  # 关闭benchmark优化

    # 设置Python哈希种子（防止哈希随机化）
    os.environ['PYTHONHASHSEED'] = str(seed)

