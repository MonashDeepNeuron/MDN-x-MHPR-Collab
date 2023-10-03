import os
import random
import sys

import numpy as np
import torch


def get_open_file_count():
    import psutil
    return len(psutil.Process().open_files())


def set_global_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__
