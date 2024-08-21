import os
import os.path as osp
from os.path import dirname, abspath
import torch
import socket
import numpy as np
import random

host_name = socket.gethostname()

DEFAULT_SEED = 42
DS_SEED = 123  # uses this seed when splitting datasets


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


set_random_seeds(DEFAULT_SEED)


# -------------- Paths
CONFIG_PATH = abspath(__file__)
SRC_ROOT = dirname(CONFIG_PATH)
PROJECT_ROOT = dirname(SRC_ROOT)
DATASET_ROOT = '/mnt/d/Datasets'

NUM_WORKERS = int(os.cpu_count() / 2)

MODEL_DIR = osp.join(SRC_ROOT, 'outputs')


# -------------- WANDB Stuff
WB_PROJECT = 'codeml-2023-emotions-challenge'
WB_ENTITY = "kacem"


AVAIL_GPUS = torch.cuda.device_count()
print('INFO AVAIL GPUS : ', AVAIL_GPUS)
