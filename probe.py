'''
Author: roy
Date: 2020-10-31 11:03:02
LastEditTime: 2020-11-01 15:25:50
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/probe.py
'''
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import jsonlines
from config import logger
from model import SelfMaskModel
from data import LAMADataset, Collator


if __name__ == "__main__":
    pass