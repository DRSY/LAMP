'''
Author: roy
Date: 2020-11-01 14:14:11
LastEditTime: 2020-11-01 14:16:53
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/model.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from typing import *


class SelfMaskModel(pl.LightningModule):
    """
    Main lightning module
    """

    def __init__(self) -> None:
        super().__init__()

    def configure_optimizers(self):
        pass

    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def forward(self, *args):
        pass
