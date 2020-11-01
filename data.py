'''
Author: roy
Date: 2020-11-01 11:08:20
LastEditTime: 2020-11-01 14:13:29
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/data.py
'''
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from typing import *

from config import (conceptNet_path, place_of_birth_path, place_of_death_path)


class LAMADataset(Dataset):
    """
    """

    def __init__(self, paths: List[str]) -> None:
        super().__init__()
        self.paths = paths
        self.datas = []

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index: int):
        return self.datas[index]


class Collator(object):
    """
    """

    def __init__(self) -> None:
        pass

    def __call__(self, data_batch: List):
        pass
