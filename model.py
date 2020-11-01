'''
Author: roy
Date: 2020-11-01 14:14:11
LastEditTime: 2020-11-01 20:07:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/model.py
'''
from inspect import getargs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from typing import *

from config import logger, get_args
from data import LAMADataset, Collator
from utils import Foobar_pruning, freeze_parameters, restore_init_state, remove_prune_reparametrization


class PruningMaskGenerator(nn.Module):
    """
    Pruning mask generator which takes as input and output a set of pruning masks for certain layers of pretrained language model
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        raise NotImplementedError


class SelfMaskModel(pl.LightningModule):
    """
    Main lightning module
    """

    def __init__(self, num_relations: int, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        # pretrained language model to be probed
        self.pretrained_language_model = AutoModelForMaskedLM.from_pretrained(
            model_name, return_dict=True)
        # a set of relation-specific pruning masking generator
        self.pruning_mask_generators = nn.ModuleList()
        for _ in range(num_relations):
            self.pruning_mask_generators.append()

    def forward(self, *args):
        pass

    def training_step(self, batch: List, batch_id: int):
        input_dict_list, labels_list, relations_in_batch = batch
        num_relations = len(relations_in_batch)
        assert len(relations_in_batch) == len(
            labels_list) == len(input_dict_list)

    def validation_step(self, *args, **kwargs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        pass


def test():
    """
    Test Utility
    """
    model = AutoModelForMaskedLM.from_pretrained(
        'bert-base-cased', return_dict=True)
    model.to(torch.device('cuda:0'))
    bert = model.bert
    num_layers = len(bert.encoder.layer)
    parameters_tobe_pruned = []
    pruning_mask_generators = []
    print(bert)
    for i in range(num_layers):
        parameters_tobe_pruned.append(
            (bert.encoder.layer[i].attention.self.query, 'weight'))
        parameters_tobe_pruned.append(
            (bert.encoder.layer[i].attention.self.key, 'weight'))
        parameters_tobe_pruned.append(
            (bert.encoder.layer[i].attention.self.value, 'weight'))
        parameters_tobe_pruned.append(
            (bert.encoder.layer[i].attention.output.dense, 'weight'))
        parameters_tobe_pruned.append(
            (bert.encoder.layer[i].intermediate.dense, 'weight'))
        parameters_tobe_pruned.append(
            (bert.encoder.layer[i].output.dense, 'weight'))
    print("Number of parameters to be pruned: {}".format(len(parameters_tobe_pruned)))
    for module, name in parameters_tobe_pruned:
        # associate each module.name with a purning mask generator matrix that has the same shape
        _size = getattr(module, name).size()
        print(_size)
        pruning_matrix = torch.nn.Parameter(torch.rand(*_size).float())
        pruning_mask_generators.append(pruning_matrix)
    nelements = 0
    for module, name in parameters_tobe_pruned:
        Foobar_pruning(module, name)
        nelements += getattr(module, name).nelement()
    print('Total number of elements: {}'.format(nelements))
    print('Pruning finished')
    print(pruning_mask_generators[0])


if __name__ == "__main__":
    test()
