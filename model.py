'''
Author: roy
Date: 2020-11-01 14:14:11
LastEditTime: 2020-11-01 21:25:24
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/model.py
'''
from inspect import getargs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from typing import *
from copy import deepcopy

from config import logger, get_args
from data import LAMADataset, Collator
from utils import Foobar_pruning, freeze_parameters, restore_init_state, remove_prune_reparametrization, bernoulli_soft_sampler, bernoulli_hard_sampler


class PruningMaskGenerator(nn.Module):
    """
    Pruning mask generator which takes as input and output a set of pruning masks for certain layers of pretrained language model
    """

    def __init__(self, shape) -> None:
        super().__init__()

    def forward(self, *args):
        raise NotImplementedError


class SelfMaskModel(pl.LightningModule):
    """
    Main lightning module
    """

    def __init__(self, num_relations: int, relation_to_id: dict, model_name: str, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_relations = num_relations
        self.relation_to_id = relation_to_id
        self.id_to_relation = {value: key for key, value in self.relation_to_id}
        self.lr = lr
        self.model_name = model_name
        # pretrained language model to be probed
        self.pretrained_language_model = AutoModelForMaskedLM.from_pretrained(
            model_name, return_dict=True)
        # a set of relation-specific pruning masking generator
        self.pruning_mask_generators = []
        for _ in range(num_relations):
            self.pruning_mask_generators.append()

        # load parameters to be pruned
        self.parameters_tobe_pruned = tuple()
        self.get_parameters_tobe_pruned()


        # create corresponding pruning mask matrics for each module and for each relation
        self.pruning_mask_generators = []
        self.create_pruning_mask_matrices()

    def get_parameters_tobe_pruned(self):
        if len(self.parameters_tobe_pruned) > 0:
            return
        num_layer = len(self.pretrained_language_model.bert.layer)
        parameters_tobe_pruned = []
        # TODO: make it more general
        bert = self.pretrained_language_model.bert
        for i in range(num_layer):
            self.parameters_tobe_pruned.append()
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
        self.parameters_tobe_pruned = tuple(parameters_tobe_pruned)

    def create_pruning_mask_matrices(self):
        for i in range(self.num_relations):
            pruning_masks = []
            for module, name in self.parameters_tobe_pruned:
                _size = getattr(module, name).size()
                pruning_mask = torch.nn.Parameter(torch.empty(*_size).float())
                pruning_mask.retain_grad()
                pruning_masks.append(pruning_mask)
            self.pruning_mask_generators.append(pruning_masks)
            


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
        all_params = []
        for ps in self.pruning_mask_generators:
            for p in ps:
                all_params.append(p)
        optimizer = optim.Adam(all_params, lr=2e-4)
        return optimizer


def test():
    """
    Test Utility
    """
    # test case
    text = "The capital of England is [MASK]."
    obj_label = "London"
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    input_dict = tokenizer(text, return_tensors='pt')
    mask_token_index = input_dict['input_ids'][0].tolist().index(
        tokenizer.mask_token_id)
    label = [-100] * len(input_dict['input_ids'][0])
    label[mask_token_index] = tokenizer.convert_tokens_to_ids([obj_label])[0]
    label = torch.tensor(label).type(input_dict['input_ids'].dtype)
    input_dict = input_dict.to(torch.device('cuda:0'))
    label = label.to(torch.device('cuda:0'))

    model = AutoModelForMaskedLM.from_pretrained(
        'bert-base-cased', return_dict=True)
    model.eval()
    freeze_parameters(model)
    model.to(torch.device('cuda:0'))
    bert = model.bert
    num_layers = len(bert.encoder.layer)
    parameters_tobe_pruned = []
    pruning_mask_generators = []
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
    print("Number of parameters to be pruned: {}".format(
        len(parameters_tobe_pruned)))
    for module, name in parameters_tobe_pruned:
        # associate each module.name with a purning mask generator matrix that has the same shape
        _size = getattr(module, name).size()
        pruning_matrix = torch.nn.Parameter(torch.rand(
            *_size).float(), requires_grad=True).to(torch.device('cuda:0'))
        pruning_matrix.retain_grad()
        pruning_mask_generators.append(pruning_matrix)
    nelements = 0
    backup_state_dict = deepcopy(model.state_dict())
    for idx, (module, name) in enumerate(parameters_tobe_pruned):
        mask = pruning_mask_generators[idx]
        Foobar_pruning(module, name, mask=mask)
        nelements += getattr(module, name).nelement()
    print('Total number of elements: {}'.format(nelements))
    print('Pruning finished')

    outputs = model(**input_dict, labels=label)
    loss = outputs.loss
    logits = outputs.logits
    print(loss)
    loss.backward()
    print(pruning_mask_generators[0].grad)
    # print(model.bert.encoder.layer[0].attention.self.query.weight)


if __name__ == "__main__":
    test()
