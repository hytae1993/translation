import torch
import numpy as np
import torch.nn as nn
import kornia.morphology as mp
import matplotlib.pyplot as plt
import os

def find_jaccard_overlap(set_1, set_2, eps=1e-5):
    """
    Find the Jaccard Overlap (IoU) of every mask between two sets of segmentation masks.
    :param set_1: set 1, a tensor of dimensions (1, 224, 224)
    :param set_2: set 2, a tensor of dimensions (1, 224, 224)
    :return: Jaccard Overlap in set 1 with respect to each of the boxes in set 2
    """
    # Find intersections
    intersection = np.logical_and(set_1, set_2)
    # Find the union
    union = np.logical_or(set_1, set_2)
    return intersection / union

def get_img(input, generator):
    with torch.no_grad():
        return generator(input).cpu()
