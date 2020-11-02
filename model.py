'''
Author: roy
Date: 2020-11-01 14:14:11
LastEditTime: 2020-11-02 16:00:57
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
from pprint import pprint

from config import logger, get_args, conceptNet_path
from data import LAMADataset, Collator, DataLoader, RandomSampler
from utils import Foobar_pruning, freeze_parameters, restore_init_state, remove_prune_reparametrization, bernoulli_soft_sampler, bernoulli_hard_sampler


class PruningMaskGenerator(nn.Module):
    """
    Pruning mask generator which takes as input and output a set of pruning masks for certain layers of pretrained language model
    """

    def __init__(self, shape) -> None:
        super().__init__()

    def forward(self, *args):
        raise NotImplementedError


class SelfMaskingModel(pl.LightningModule):
    """
    Main lightning module
    """

    def __init__(self, num_relations: int, relation_to_id: dict, model_name: str, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_relations = num_relations
        self.relation_to_id = relation_to_id
        self.id_to_relation = {value: key for key,
                               value in self.relation_to_id.items()}
        self.lr = lr
        self.model_name = model_name
        # pretrained language model to be probed
        self.pretrained_language_model = AutoModelForMaskedLM.from_pretrained(
            model_name, return_dict=True)

        # load parameters to be pruned
        self.parameters_tobe_pruned = tuple()
        self.get_parameters_tobe_pruned()

        # create corresponding pruning mask matrics for each module and for each relation
        self.pruning_mask_generators = []
        self.create_pruning_mask_matrices()
        self.init_pruning_masks(torch.nn.init.uniform)

        # create copy of init state
        self.orig_state_dict = deepcopy(
            self.pretrained_language_model.state_dict())

    def get_parameters_tobe_pruned(self):
        if len(self.parameters_tobe_pruned) > 0:
            return
        num_layer = len(self.pretrained_language_model.bert.encoder.layer)
        parameters_tobe_pruned = []
        # TODO: make it more general
        bert = self.pretrained_language_model.bert
        for i in range(num_layer):
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
        for _ in range(self.num_relations):
            pruning_masks = []
            for module, name in self.parameters_tobe_pruned:
                _size = getattr(module, name).size()
                pruning_mask = torch.nn.Parameter(torch.empty(*_size).float())
                pruning_mask.retain_grad()
                pruning_masks.append(pruning_mask)
            self.pruning_mask_generators.append(pruning_masks)

    def init_pruning_masks(self, init_method: Callable):
        for ps in self.pruning_mask_generators:
            for p in ps:
                init_method(p)

    def move_pruning_mask_generators(self, device):
        for ps in self.pruning_mask_generators:
            for i in range(len(ps)):
                ps[i] = ps[i].to(device)

    def forward(self, input_dict, labels):
        outputs = self.pretrained_language_model(**input_dict, labels=labels)
        loss = outputs.loss
        return loss

    def prune(self, pruning_masks):
        for pruning_mask, (module, name) in zip(pruning_masks, self.parameters_tobe_pruned):
            Foobar_pruning(module, name, pruning_mask)

    def restore(self):
        for module, name in self.parameters_tobe_pruned:
            prune.remove(module, name)
        restore_init_state(self.pretrained_language_model,
                           self.orig_state_dict)

    def feed_batch(self, input_dict, labels, relation_id: int, device):
        """
        feed a batch of input with the same relation
        """
        pruning_masks_logits = self.pruning_mask_generators[relation_id]
        pruning_masks_soft_samples = []
        for pruning_mask_logits in pruning_masks_logits:
            soft_sample = bernoulli_soft_sampler(
                pruning_mask_logits.to(device), temperature=0.1)
            pruning_masks_soft_samples.append(soft_sample)
        self.prune(pruning_masks_soft_samples)
        # feed input batch and backward loss
        loss = self(input_dict, labels)
        loss.backward()
        self.restore()
        return loss.detach().item()

    def training_step(self, batch: List, batch_id: int):
        input_dict_list, labels_list, relations_in_batch = batch
        num_relations = len(relations_in_batch)
        assert len(relations_in_batch) == len(
            labels_list) == len(input_dict_list)
        total_loss = .0
        for i in range(len(relations_in_batch)):
            relation_id = relations_in_batch[i]
            pruning_masks = self.pruning_mask_generators[relation_id]
            self.prune(pruning_masks)
            # feed examples
            input_dict = input_dict_list[i]
            labels = labels_list[i]
            loss = self(input_dict, labels)
            total_loss += loss

        return {'loss': total_loss}

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
        optimizer = optim.Adam(all_params, lr=self.hparams.lr)
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
    cp_pruning_mask_generators = []
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
        pruning_matrix = torch.nn.Parameter(
            torch.rand(*_size)).to(torch.device('cuda:0'))
        pruning_matrix.retain_grad()
        pruning_mask_generators.append(pruning_matrix)
    # opt = optim.Adam(pruning_mask_generators, lr=2e-4)
    backup_state_dict = deepcopy(model.state_dict())
    for idx, (module, name) in enumerate(parameters_tobe_pruned):
        mask = pruning_mask_generators[idx]
        Foobar_pruning(module, name, mask=mask)
    print('Pruning finished')

    outputs = model(**input_dict, labels=label)
    loss = outputs.loss
    print(pruning_mask_generators[0].grad)
    print(pruning_mask_generators[1].grad)
    loss.backward()
    print('loss backward')
    print(pruning_mask_generators[0].grad)
    print(pruning_mask_generators[1].grad)


if __name__ == "__main__":
    pass
