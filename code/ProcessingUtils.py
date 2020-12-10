"""
ProcessingUtils.py
Written by Adam Lavertu
Stanford University
"""

import os

import torch
import numpy as np
from models import CheckNet


def check_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_model(chk_pth):
    temp_model = CheckNet()
    _ = temp_model.load_state_dict(
        torch.load(chk_pth, map_location=torch.device("cpu"))
    )
    _ = temp_model.eval()
    return temp_model


def prep_image_data(images, transforms):
    out_tensors = []
    for temp_im in images:
        temp_im = temp_im.astype(np.float32)
        temp_im = 1.0 - temp_im
        out_tensors.append(transforms(temp_im))
    return torch.stack(out_tensors)
