'''
Author: roy
Date: 2020-11-01 14:14:11
LastEditTime: 2020-11-09 15:15:39
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/model.py
'''
from copy import deepcopy
from pprint import pprint
from typing import *
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.optim as optim
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from utils import (Foobar_pruning, bernoulli_hard_sampler,
                   bernoulli_soft_sampler, freeze_parameters,
                   remove_prune_reparametrization, restore_init_state)


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
    init_methods = {
        'uniform': torch.nn.init.uniform_,
        'normal': partial(torch.nn.init.normal_, mean=0, std=1),
        'zeros': torch.nn.init.zeros_, # 50 % initial sparsity
        '0.41': partial(torch.nn.init.constant_, val=0.41), # 40% initial sparsity
        '0.62': partial(torch.nn.init.constant_, val=0.62), # 35% initial sparsity
        '0.85': partial(torch.nn.init.constant_, val=0.85), # 30% initial sparsity
        'ones': torch.nn.init.ones_, # 27% initial sparsity
        '1.38': partial(torch.nn.init.constant_, val=1.38), # 20% initial sparsity
        '2.75': partial(torch.nn.init.constant_, val=2.75), # 6% initial sparsity
        '2.95': partial(torch.nn.init.constant_, val=2.95), # 5% initial sparsity
    }

    def __init__(self, bli: int, tli: int, num_relations: int, relation_to_id: dict, model_name: str, lr: float, init_method: str) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.bli = bli
        self.tli = tli
        self.num_relations = num_relations
        self.relation_to_id = relation_to_id
        self.id_to_relation = {value: key for key,
                               value in self.relation_to_id.items()}
        print("Relations:")
        pprint(self.relation_to_id)
        self.lr = lr
        self.model_name = model_name
        # pretrained language model to be probed
        self.pretrained_language_model = AutoModelForMaskedLM.from_pretrained(
            model_name, return_dict=True, output_hidden_states=True, output_attentions=True)

        # load parameters to be pruned
        self.parameters_tobe_pruned = tuple()
        self.get_parameters_tobe_pruned(bli, tli)

        # create corresponding pruning mask matrics for each module and for each relation
        self.pruning_mask_generators = []
        self.create_pruning_mask_matrices()
        self.init_pruning_masks(self.init_methods[init_method])

        # create copy of init state
        self.orig_state_dict = deepcopy(
            self.pretrained_language_model.state_dict())

    def get_parameters_tobe_pruned(self, bli, tli):
        if len(self.parameters_tobe_pruned) > 0:
            return
        parameters_tobe_pruned = []
        if 'albert' in self.model_name:
            layers = self.pretrained_language_model.albert.encoder.albert_layer_groups[0].albert_layers[0]
        elif 'roberta' in self.model_name:
            layers = self.pretrained_language_model.roberta.encoder.layer
        elif 'distil' in self.model_name:
            layers = self.pretrained_language_model.distilbert.transformer.layer
        elif 'bert' in self.model_name:
            layers = self.pretrained_language_model.bert.encoder.layer
        elif 'mpnet' in self.model_name:
            layers = self.pretrained_language_model.mpnet.encoder.layer
        elif 'electra' in self.model_name:
            layers = self.pretrained_language_model.electra.encoder.layer
        if 'albert' in self.model_name:
            parameters_tobe_pruned.append((layers.attention.query, 'weight'))
            parameters_tobe_pruned.append((layers.attention.key, 'weight'))
            parameters_tobe_pruned.append((layers.attention.value, 'weight'))
            parameters_tobe_pruned.append((layers.attention.dense, 'weight'))
            parameters_tobe_pruned.append((layers.ffn, 'weight'))
            parameters_tobe_pruned.append((layers.ffn_output, 'weight'))
        elif 'mpnet' in self.model_name:
            for i in range(bli, tli+1):
                parameters_tobe_pruned.append(
                    (layers[i].attention.attn.q, 'weight'))
                parameters_tobe_pruned.append(
                    (layers[i].attention.attn.k, 'weight'))
                parameters_tobe_pruned.append(
                    (layers[i].attention.attn.v, 'weight'))
                parameters_tobe_pruned.append(
                    (layers[i].attention.attn.o, 'weight'))
                parameters_tobe_pruned.append(
                    (layers[i].intermediate.dense, 'weight'))
                parameters_tobe_pruned.append(
                    (layers[i].output.dense, 'weight'))
        else:
            for i in range(bli, tli+1):
                try:
                    parameters_tobe_pruned.append(
                        (layers[i].attention.self.query, 'weight'))
                    parameters_tobe_pruned.append(
                        (layers[i].attention.self.key, 'weight'))
                    parameters_tobe_pruned.append(
                        (layers[i].attention.self.value, 'weight'))
                    parameters_tobe_pruned.append(
                        (layers[i].attention.output.dense, 'weight'))
                    parameters_tobe_pruned.append(
                        (layers[i].intermediate.dense, 'weight'))
                    parameters_tobe_pruned.append(
                        (layers[i].output.dense, 'weight'))
                except Exception:
                    parameters_tobe_pruned.append(
                        (layers[i].attention.q_lin, 'weight')
                    )
                    parameters_tobe_pruned.append(
                        (layers[i].attention.k_lin, 'weight')
                    )
                    parameters_tobe_pruned.append(
                        (layers[i].attention.v_lin, 'weight')
                    )
                    parameters_tobe_pruned.append(
                        (layers[i].attention.out_lin, 'weight')
                    )
                    parameters_tobe_pruned.append(
                        (layers[i].ffn.lin1, 'weight')
                    )
                    parameters_tobe_pruned.append(
                        (layers[i].ffn.lin2, 'weight')
                    )
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

    @staticmethod
    def get_position_of_gold_label(logits, labels):
        bs = labels.size(0)
        positions = []
        for i in range(bs):
            tmp = labels[i].eq(-100).eq(0).int().tolist()
            idx = tmp.index(1)
            mask_token_id = labels[i][idx].item()
            _sorted = torch.sort(
                logits[i, idx], descending=True).indices.tolist()
            position = _sorted.index(mask_token_id)
            positions.append(position)
        return positions

    def forward(self, input_dict, labels, rl: bool = False):
        outputs = self.pretrained_language_model(**input_dict, labels=labels)
        loss = outputs.loss
        if not rl:
            return loss
        logits = outputs.logits
        positions = self.get_position_of_gold_label(logits, labels)
        return positions

    def prune(self, pruning_masks):
        for pruning_mask, (module, name) in zip(pruning_masks, self.parameters_tobe_pruned):
            Foobar_pruning(module, name, pruning_mask)

    def restore(self):
        for module, name in self.parameters_tobe_pruned:
            prune.remove(module, name)
        restore_init_state(self.pretrained_language_model,
                           self.orig_state_dict)

    @torch.no_grad()
    def get_cls_representation(self, input_dict, relation_id: int, device, use_fullscale=False):
        if not use_fullscale:
            pruning_masks_logits = self.pruning_mask_generators[relation_id]
            pruning_masks_soft_samples = []
            for pruning_mask_logits in pruning_masks_logits:
                cuda_mask = pruning_mask_logits.to(device)
                _probs = torch.sigmoid(cuda_mask)
                _probs[_probs>0.5] = 1
                _probs[_probs<=0.5] = 0
                pruning_masks_soft_samples.append(_probs)
            self.prune(pruning_masks=pruning_masks_soft_samples)

        outputs = self.pretrained_language_model(**input_dict)
        hidden_states = outputs.hidden_states
        cls_representations = hidden_states[-1][:, 0, :] # (batch_size, hidden_dim)
        if not use_fullscale:
            self.restore()
        return cls_representations

    @torch.no_grad()
    def get_mask_representation(self, input_dict, relation_id: int, mask_index, device, use_fullscale=False):
        if not use_fullscale:
            pruning_masks_logits = self.pruning_mask_generators[relation_id]
            pruning_masks_soft_samples = []
            for pruning_mask_logits in pruning_masks_logits:
                cuda_mask = pruning_mask_logits.to(device)
                _probs = torch.sigmoid(cuda_mask)
                _probs[_probs>0.5] = 1
                _probs[_probs<=0.5] = 0
                pruning_masks_soft_samples.append(_probs)
            self.prune(pruning_masks=pruning_masks_soft_samples)

        outputs = self.pretrained_language_model(**input_dict)
        hidden_states = outputs.hidden_states
        cls_representations = hidden_states[-1][:, mask_index, :] # (batch_size, hidden_dim)
        if not use_fullscale:
            self.restore()
        return cls_representations

    def feed_batch(self, input_dict, labels, relation_id: int, device):
        """
        feed a batch of input with the same relation
        use soft approximation of discrete Bernoulli distribution
        """
        pruning_masks_logits = self.pruning_mask_generators[relation_id]
        pruning_masks_soft_samples = []
        for pruning_mask_logits in pruning_masks_logits:
            soft_sample = bernoulli_soft_sampler(
                pruning_mask_logits.to(device), temperature=0.1)
            pruning_masks_soft_samples.append(soft_sample)
        self.prune(pruning_masks=pruning_masks_soft_samples)

        # feed input batch and backward loss
        loss = self(input_dict, labels)
        loss.backward()
        clip_grad_norm_(pruning_masks_logits, max_norm=5)
        self.restore()
        return loss.detach().item()

    def feed_batch_straight_through(self, input_dict, labels, relation_id: int, device):
        """
        feed a batch of input with the same relation
        use hard straight-through gradient estimator
        """
        pruning_masks_logits = self.pruning_mask_generators[relation_id]
        pruning_masks_soft_samples = []
        for pruning_mask_logits in pruning_masks_logits:
            cuda_mask = pruning_mask_logits.to(device)
            _probs = torch.sigmoid(cuda_mask)
            _probs[_probs>0.5] = 1
            _probs[_probs<=0.5] = 0
            straight_through_sample = (_probs - cuda_mask).detach() + cuda_mask
            pruning_masks_soft_samples.append(straight_through_sample)
        self.prune(pruning_masks=pruning_masks_soft_samples)

        # feed input batch and backward loss
        loss = self(input_dict, labels)
        loss.backward()
        clip_grad_norm_(pruning_masks_logits, max_norm=5)
        self.restore()
        return loss.detach().item()

    def feed_batch_rl(self, input_dict, labels, relation_id, device):
        """
        feed a batch of input with the same relation
        use hard sampling of discrete Bernoulli distribution
        NOTE: potentially not as effective as soft approximation
        """
        pruning_masks_logits = self.pruning_mask_generators[relation_id]
        pruning_masks_hard_samples = []
        for pruning_mask_logits in pruning_masks_logits:
            hard_sample, log_prob = bernoulli_hard_sampler(
                torch.sigmoid(pruning_mask_logits))
            pruning_masks_hard_samples.append(hard_sample)
        self.prune(pruning_masks=pruning_masks_hard_samples)

        # feed input batch and backward loss
        positions = self(input_dict, labels, rl=True)

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
