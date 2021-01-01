'''
Author: roy
Date: 2020-10-31 11:03:02
LastEditTime: 2020-11-10 10:22:58
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/probe.py
'''
from functools import reduce
from itertools import combinations
import pickle
import json
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import os
import jsonlines
from pprint import pprint
from tqdm import tqdm
import pytorch_lightning as pl
from typing import *
import prettytable as pt
from copy import deepcopy
from nltk.corpus import stopwords
import numpy as np
import datasets
from datasets import load_dataset
copa_dataset = datasets.load_from_disk("./copa_train")
stop_words = set(stopwords.words('english'))


from config import logger, get_args, TRANSFORMER_LAYERS
from model import SelfMaskingModel
from data import LAMADataset, Collator, DataLoader, RandomSampler
import utils
from utils import freeze_parameters


def test_pl(args):
    # set random seed
    seed = args.seed
    pl.seed_everything(seed)

    device = torch.device('cuda:0')
    toy_dataset = LAMADataset(args.data_path)
    relation_to_id = toy_dataset.relation_to_id
    logger.info("Relations for {}".format(args.data_path))
    pprint(relation_to_id)
    collator = Collator(relation_to_id, args.model_name, args.max_length)
    toy_dataloader = DataLoader(
        toy_dataset, collate_fn=collator, batch_size=20, sampler=RandomSampler(toy_dataset))
    pl_model = SelfMaskingModel(
        len(relation_to_id), relation_to_id, args.model_name, args.lr)
    pl_model.to(device)
    all_params = []
    for ps in pl_model.pruning_mask_generators:
        for p in ps:
            all_params.append(p)
    optimizer = optim.Adam(all_params, lr=args.lr)
    linear_warmup_decay_scheduler = get_linear_schedule_with_warmup(
        optimizer, args.warmup*args.max_steps, args.max_steps)
    for b in toy_dataloader:
        optimizer.zero_grad()
        for i in range(len(b[-1])):
            relation_id = b[-1][i]
            input_dict = b[0][i].to(device)
            labels = b[1][i].to(device)
            loss = pl_model.feed_batch(input_dict, labels, relation_id, device)
            logger.info("loss for relation {}: {}".format(
                pl_model.id_to_relation[relation_id], loss))
        logger.info(
            "Finish updating all pruning mask generators for a mini-batch")
        optimizer.step()
        linear_warmup_decay_scheduler.step()
        exit()


@torch.no_grad()
def prediction(model: SelfMaskingModel, tokenizer, device, masked_sentence, relation_id):
    if relation_id != None:
        pruning_mask_generator = model.pruning_mask_generators[relation_id]
        hard_samples = []
        for pruning_mask in pruning_mask_generator:
            hard_sample = torch.sigmoid(pruning_mask).to(device)
            hard_sample[hard_sample>0.5] = 1
            hard_sample[hard_sample<=0.5] = 0
            hard_samples.append(hard_sample)
        model.prune(pruning_masks=hard_samples)
    input_dict = tokenizer(masked_sentence, return_tensors='pt').to(device)
    mask_index = input_dict['input_ids'][0].tolist().index(tokenizer.mask_token_id)
    backbone_model = model.pretrained_language_model
    outputs = backbone_model(**input_dict)
    logits = outputs.logits
    prob = torch.softmax(logits[0, mask_index, :], dim=-1)
    values, indices = torch.topk(prob, k=5)
    cnt = 0
    for token in tokenizer.decode(indices).strip().split():
        print("{}: {}".format(token, values[cnt]))
        cnt += 1
    if relation_id != None:
        model.restore()


@torch.no_grad()
def probing_losses(corpus_file_path: str, model: SelfMaskingModel, collator: Collator, device: torch.device, relation: str, total: int):
    corpus_fileobj = open(corpus_file_path, 'r')
    tokenizer = collator.tokenizer
    if relation != None:
        pruning_mask_generator = model.pruning_mask_generators[model.relation_to_id[relation]]
        hard_samples = []
        for pruning_mask in pruning_mask_generator:
            hard_sample = torch.sigmoid(pruning_mask).to(device)
            hard_sample[hard_sample>0.5] = 1
            hard_sample[hard_sample<=0.5] = 0
            hard_samples.append(hard_sample)
        model.prune(pruning_masks=hard_samples)
        print("Model pruned with mask for relation: {}".format(model.relation_to_id[relation]))
    losses = []
    for instance in tqdm(jsonlines.Reader(corpus_fileobj), total=total):
        masked_sentences = instance['masked_sentences']
        obj_label = instance['obj_label'].lower()
        input_dict = tokenizer(masked_sentences[0], return_tensors='pt').to(device)
        labels = collator.get_label(input_dict['input_ids'][0].tolist(), obj_label)
        labels = torch.tensor(labels).type(input_dict['input_ids'].dtype).to(device)
        loss = model.forward(input_dict, labels, rl=False).detach().to('cpu').item()
        losses.append(loss)
    avg_losses = sum(losses) / len(losses)
    return avg_losses


@torch.no_grad()
def save_cls_representations(model: SelfMaskingModel, tokenizer, device, corpus_file_path: str, total: int, embedding_save_pth: str, relation_ids_save_pth: str, use_fullscale: bool):
    corpus_fileobj = open(corpus_file_path, mode='r', encoding='utf-8')
    vectors = []
    relation_ids = []
    for instance in tqdm(jsonlines.Reader(corpus_fileobj), total=total):
        total += 1
        if 'TREx' in corpus_file_path:
            evidences = instance['evidences']
            obj_label = instance['obj_label'].lower()
            predicate_id = instance['predicate_id']
            relation_id = model.relation_to_id[predicate_id]
            masked_sentence = evidences[0]['masked_sentence']
            complete_sentence = masked_sentence.replace('[MASK]', obj_label)
            input_dict = tokenizer(complete_sentence, return_tensors='pt').to(device)
            cls_vector = model.get_cls_representation(input_dict, relation_id, device, use_fullscale=use_fullscale)[0].to('cpu').numpy()
            vectors.append(cls_vector)
            relation_ids.append(relation_id)
        else:
            masked_sentences = instance['masked_sentences']
            obj_label = instance['obj_label'].lower()
            complete_sentence = masked_sentences[0].replace("[MASK]", obj_label)
            relation = instance['pred']
            relation_id = model.relation_to_id[relation]
            input_dict = tokenizer(complete_sentence, return_tensors='pt').to(device)
            cls_vector = model.get_cls_representation(input_dict, relation_id, device, use_fullscale=use_fullscale)[0].to('cpu').numpy()
            vectors.append(cls_vector)
            relation_ids.append(relation_id)
    vectors = np.array(vectors)
    corpus_fileobj.close()
    with open(embedding_save_pth, 'wb') as f:
        pickle.dump(vectors,f)
    with open(relation_ids_save_pth, 'wb') as f:
        pickle.dump(relation_ids, f)
    return vectors, relation_ids

@torch.no_grad()
def save_mask_representations(model: SelfMaskingModel, tokenizer, device, corpus_file_path: str, total: int, embedding_save_pth: str, relation_ids_save_pth: str, use_fullscale: bool):
    corpus_fileobj = open(corpus_file_path, mode='r', encoding='utf-8')
    vectors = []
    relation_ids = []
    for instance in tqdm(jsonlines.Reader(corpus_fileobj), total=total):
        total += 1
        if 'TREx' in corpus_file_path:
            evidences = instance['evidences']
            obj_label = instance['obj_label'].lower()
            predicate_id = instance['predicate_id']
            relation_id = model.relation_to_id[predicate_id]
            masked_sentence = evidences[0]['masked_sentence']
            complete_sentence = masked_sentence.replace('[MASK]', obj_label)
            input_dict = tokenizer(complete_sentence, return_tensors='pt').to(device)
            cls_vector = model.get_mask_representation(input_dict, relation_id, device, use_fullscale=use_fullscale)[0].to('cpu').numpy()
            vectors.append(cls_vector)
            relation_ids.append(relation_id)
        else:
            masked_sentences = instance['masked_sentences']
            obj_label = instance['obj_label'].lower()
            relation = instance['pred']
            relation_id = model.relation_to_id[relation]
            mask_input_dict = tokenizer(masked_sentences[0], return_tensors='pt').to(device)
            mask_index = mask_input_dict['input_ids'][0].to('cpu').tolist().index(tokenizer.mask_token_id)
            mask_vector = model.get_mask_representation(mask_input_dict, relation_id, mask_index, device, use_fullscale=use_fullscale)[0].to('cpu').numpy()
            vectors.append(mask_vector)
            relation_ids.append(relation_id)
    vectors = np.array(vectors)
    corpus_fileobj.close()
    with open(embedding_save_pth, 'wb') as f:
        pickle.dump(vectors,f)
    with open(relation_ids_save_pth, 'wb') as f:
        pickle.dump(relation_ids, f)
    return vectors, relation_ids

@torch.no_grad()
def validate(model: SelfMaskingModel, tokenizer, device, corpus_file_path: str, total: int, use_expectation: bool = True):
    """
    validate pruning masks on LAMA dataset
    """
    corpus_fileobj = open(corpus_file_path, mode='r', encoding='utf-8')
    num_relations = len(model.pruning_mask_generators)
    relation_specific_p1 = [.0] * num_relations
    relation_specific_p2 = [.0] * num_relations
    relation_specific_p3 = [.0] * num_relations
    relation_specific_p5 = [.0] * num_relations
    relation_specific_total = [0] * num_relations
    relation_specific_unpruned_p1 = [.0] * num_relations
    relation_specific_unpruned_p2 = [.0] * num_relations
    relation_specific_unpruned_p3 = [.0] * num_relations
    relation_specific_unpruned_p5 = [.0] * num_relations
    pruned_p1 = .0
    pruned_top1 = 0
    unpruned_p1 = .0
    unpruned_top1 = 0
    comparison = []
    for instance in tqdm(jsonlines.Reader(corpus_fileobj), total=total):
        total += 1
        if 'TREx' in corpus_file_path:
            evidences = instance['evidences']
            obj_label = instance['obj_label'].lower()
            predicate_id = instance['predicate_id']
            relation_id = model.relation_to_id[predicate_id]
            masked_sentence = evidences[0]['masked_sentence']
            pruning_mask_generator = model.pruning_mask_generators[relation_id]
            hard_samples = []
            for pruning_mask in pruning_mask_generator:
                if use_expectation:
                    hard_sample = torch.sigmoid(pruning_mask).to(
                        device)  # use the expectation values
                else:
                    hard_sample = torch.sigmoid(pruning_mask).to(device)
                    hard_sample[hard_sample>0.5] = 1
                    hard_sample[hard_sample<=0.5] = 0
                hard_samples.append(hard_sample)
            model.prune(pruning_masks=hard_samples)
            pruned_predictions = utils.LAMA(
                model.pretrained_language_model, tokenizer, device, masked_sentence, topk=5)
            try:
                pos = pruned_predictions.index(obj_label)
                if pos == 0:
                    pruned_top1 += 1
                    relation_specific_p1[relation_id] += 1
                if pos in [0, 1]:
                    relation_specific_p2[relation_id] += 1
                if pos in [0, 1, 2]:
                    relation_specific_p3[relation_id] += 1
            except ValueError:
                pass
            model.restore()
            unpruned_predictions = utils.LAMA(
                model.pretrained_language_model, tokenizer, device, masked_sentence, topk=5)
            try:
                pos = unpruned_predictions.index(obj_label)
                if pos == 0:
                    unpruned_top1 += 1
                    relation_specific_unpruned_p1[relation_id] += 1
                if pos in [0, 1]:
                    relation_specific_unpruned_p2[relation_id] += 1
                if pos in [0, 1, 2]:
                    relation_specific_unpruned_p3[relation_id] += 1
            except ValueError:
                pass
        else:
            flag = False
            masked_sentences = instance['masked_sentences']
            obj_label = instance['obj_label'].lower()
            relation = instance['pred']
            relation_id = model.relation_to_id[relation]
            relation_specific_total[relation_id] += 1
            pruning_mask_generator = model.pruning_mask_generators[relation_id]
            hard_samples = []
            for pruning_mask in pruning_mask_generator:
                if use_expectation:
                    hard_sample = torch.sigmoid(pruning_mask).to(
                        device)  # use the expectation values
                else:
                    hard_sample = torch.sigmoid(pruning_mask).to(device)
                    hard_sample[hard_sample>0.5] = 1
                    hard_sample[hard_sample<=0.5] = 0
                hard_samples.append(hard_sample)
            model.prune(pruning_masks=hard_samples)
            pruned_predictions = utils.LAMA(
                model.pretrained_language_model, tokenizer, device, masked_sentences[0], topk=5)
            try:
                pos = pruned_predictions.index(obj_label)
                if pos == 0:
                    flag = True
                    pruned_top1 += 1
                    relation_specific_p1[relation_id] += 1
                if pos in [0, 1]:
                    relation_specific_p2[relation_id] += 1
                if pos in [0, 1, 2]:
                    relation_specific_p3[relation_id] += 1
                if pos in [0, 1, 2, 3, 4]:
                    relation_specific_p5[relation_id] += 1
            except ValueError:
                pass
            model.restore()
            unpruned_predictions = utils.LAMA(
                model.pretrained_language_model, tokenizer, device, masked_sentences[0].replace('[MASK]', tokenizer.mask_token), topk=5)
            # print(pruned_predictions)
            # print(unpruned_predictions)
            try:
                pos = unpruned_predictions.index(obj_label)
                if pos == 0:
                    flag = False
                    unpruned_top1 += 1
                    relation_specific_unpruned_p1[relation_id] += 1
                if pos in [0, 1]:
                    relation_specific_unpruned_p2[relation_id] += 1
                if pos in [0, 1, 2]:
                    relation_specific_unpruned_p3[relation_id] += 1
                if pos in [0, 1, 2, 3, 4]:
                    relation_specific_unpruned_p5[relation_id] += 1
            except ValueError:
                pass
            # if flag:
               # comparison.append([{'masked_sentence': masked_sentences[0], 'relation': relation, 'output_pruned': pruned_predictions, 'output_fullscale': unpruned_predictions}])
    # sparsity
    sparsity_dict = utils.sparsity(model, args.init_method)

    # macro-average
    pruned_p1 = pruned_top1 / total
    unpruned_p1 = unpruned_top1 / total

    # micro-average
    for i in range(num_relations):
        relation_specific_p1[i] = relation_specific_p1[i] / \
            relation_specific_total[i]
        relation_specific_p2[i] = relation_specific_p2[i] / \
            relation_specific_total[i]
        relation_specific_p3[i] = relation_specific_p3[i] / \
            relation_specific_total[i]
        relation_specific_p5[i] = relation_specific_p5[i] / \
            relation_specific_total[i]
        relation_specific_unpruned_p1[i] = relation_specific_unpruned_p1[i] / \
            relation_specific_total[i]
        relation_specific_unpruned_p2[i] = relation_specific_unpruned_p2[i] / \
            relation_specific_total[i]
        relation_specific_unpruned_p3[i] = relation_specific_unpruned_p3[i] / \
            relation_specific_total[i]
        relation_specific_unpruned_p5[i] = relation_specific_unpruned_p5[i] / \
            relation_specific_total[i]

    corpus_fileobj.close()
    ret_dict = {}
    ret_dict['macro_pruned_p1'] = pruned_p1
    ret_dict['macro_unpruned_p1'] = unpruned_p1
    ret_dict['micro_pruned_p1'] = sum(
        relation_specific_p1) / len(relation_specific_p1)
    ret_dict['micro_pruned_p2'] = sum(
        relation_specific_p2) / len(relation_specific_p2)
    ret_dict['micro_pruned_p3'] = sum(
        relation_specific_p3) / len(relation_specific_p3)
    ret_dict['micro_pruned_p5'] = sum(
        relation_specific_p5) / len(relation_specific_p5)
    ret_dict['micro_unpruned_p1'] = sum(
        relation_specific_unpruned_p1) / len(relation_specific_unpruned_p1)
    ret_dict['micro_unpruned_p2'] = sum(
        relation_specific_unpruned_p2) / len(relation_specific_unpruned_p2)
    ret_dict['micro_unpruned_p3'] = sum(
        relation_specific_unpruned_p3) / len(relation_specific_unpruned_p3)
    ret_dict['micro_unpruned_p5'] = sum(
        relation_specific_unpruned_p5) / len(relation_specific_unpruned_p5)
    ret_dict['sparsity'] = sparsity_dict
    P1_dict = {}
    P1_dict_fullscale = {}
    for i in range(num_relations):
        P1_dict[model.id_to_relation[i]] = relation_specific_p1[i]
    for i in range(num_relations):
        P1_dict_fullscale[model.id_to_relation[i]] = relation_specific_unpruned_p1[i]
    print("Pruned:")
    for relation in P1_dict:
        print("{}: {:.3f}".format(relation, P1_dict[relation]))
    print("Unpruned:")
    for relation in P1_dict_fullscale:
        print("{}: {:.3f}".format(relation, P1_dict_fullscale[relation]))
    # if len(comparison) > 0:
    #     with open("Comparison_generation.json", mode='w', encoding='utf-8') as f:
    #         json.dump(comparison, f)
    return ret_dict


@torch.no_grad()
def validate_mismatched(model: SelfMaskingModel, tokenizer, device, corpus_file_path: str, total: int, use_expectation: bool = True):
    """
    apply mismatched mask for testing transferability of relation-specific subnetwork
    """
    corpus_fileobj = open(corpus_file_path, mode='r', encoding='utf-8')
    num_relations = len(model.pruning_mask_generators)
    relation_specific_p1 = [.0] * num_relations
    relation_specific_p2 = [.0] * num_relations
    relation_specific_p3 = [.0] * num_relations
    relation_specific_total = [0] * num_relations
    relation_specific_unpruned_p1 = [.0] * num_relations
    relation_specific_unpruned_p2 = [.0] * num_relations
    relation_specific_unpruned_p3 = [.0] * num_relations
    pruned_p1 = .0
    pruned_top1 = 0
    unpruned_p1 = .0
    unpruned_top1 = 0
    for instance in tqdm(jsonlines.Reader(corpus_fileobj), total=total):
        total += 1
        if 'TREx' in corpus_file_path:
            evidences = instance['evidences']
            obj_label = instance['obj_label'].lower()
            predicate_id = instance['predicate_id']
            relation_id = model.relation_to_id[predicate_id]
            masked_sentence = evidences[0]['masked_sentence']
            pruning_mask_generator = model.pruning_mask_generators[relation_id]
            hard_samples = []
            for pruning_mask in pruning_mask_generator:
                if use_expectation:
                    hard_sample = torch.sigmoid(pruning_mask).to(
                        device)  # use the expectation values
                else:
                    hard_sample = torch.sigmoid(pruning_mask).to(device)
                    hard_sample[hard_sample>0.5] = 1
                    hard_sample[hard_sample<=0.5] = 0
                hard_samples.append(hard_sample)
            model.prune(pruning_masks=hard_samples)
            pruned_predictions = utils.LAMA(
                model.pretrained_language_model, tokenizer, device, masked_sentence, topk=5)
            try:
                pos = pruned_predictions.index(obj_label)
                if pos == 0:
                    pruned_top1 += 1
                    relation_specific_p1[relation_id] += 1
                if pos in [0, 1]:
                    relation_specific_p2[relation_id] += 1
                if pos in [0, 1, 2]:
                    relation_specific_p3[relation_id] += 1
            except ValueError:
                pass
            model.restore()
            unpruned_predictions = utils.LAMA(
                model.pretrained_language_model, tokenizer, device, masked_sentence, topk=5)
            try:
                pos = unpruned_predictions.index(obj_label)
                if pos == 0:
                    unpruned_top1 += 1
                    relation_specific_unpruned_p1[relation_id] += 1
                if pos in [0, 1]:
                    relation_specific_unpruned_p2[relation_id] += 1
                if pos in [0, 1, 2]:
                    relation_specific_unpruned_p3[relation_id] += 1
            except ValueError:
                pass
        else:
            masked_sentences = instance['masked_sentences']
            obj_label = instance['obj_label'].lower()
            relation = instance['pred']
            relation_id = model.relation_to_id[relation]
            shuffled_relation_id = (relation_id + 15) % len(model.relation_to_id)
            relation_specific_total[relation_id] += 1
            pruning_mask_generator = model.pruning_mask_generators[shuffled_relation_id]
            hard_samples = []
            for pruning_mask in pruning_mask_generator:
                if use_expectation:
                    hard_sample = torch.sigmoid(pruning_mask).to(
                        device)  # use the expectation values
                else:
                    hard_sample = torch.sigmoid(pruning_mask).to(device)
                    hard_sample[hard_sample>0.5] = 1
                    hard_sample[hard_sample<=0.5] = 0
                hard_samples.append(hard_sample)
            model.prune(pruning_masks=hard_samples)
            pruned_predictions = utils.LAMA(
                model.pretrained_language_model, tokenizer, device, masked_sentences[0], topk=5)
            try:
                pos = pruned_predictions.index(obj_label)
                if pos == 0:
                    pruned_top1 += 1
                    relation_specific_p1[relation_id] += 1
                if pos in [0, 1]:
                    relation_specific_p2[relation_id] += 1
                if pos in [0, 1, 2]:
                    relation_specific_p3[relation_id] += 1
            except ValueError:
                pass
            model.restore()
            unpruned_predictions = utils.LAMA(
                model.pretrained_language_model, tokenizer, device, masked_sentences[0].replace('[MASK]', tokenizer.mask_token), topk=5)
            # print(pruned_predictions)
            # print(unpruned_predictions)
            try:
                pos = unpruned_predictions.index(obj_label)
                if pos == 0:
                    unpruned_top1 += 1
                    relation_specific_unpruned_p1[relation_id] += 1
                if pos in [0, 1]:
                    relation_specific_unpruned_p2[relation_id] += 1
                if pos in [0, 1, 2]:
                    relation_specific_unpruned_p3[relation_id] += 1
            except ValueError:
                pass
    # sparsity
    sparsity_dict = utils.sparsity(model, args.init_method)

    # macro-average
    pruned_p1 = pruned_top1 / total
    unpruned_p1 = unpruned_top1 / total

    # micro-average
    for i in range(num_relations):
        relation_specific_p1[i] = relation_specific_p1[i] / \
            relation_specific_total[i]
        relation_specific_p2[i] = relation_specific_p2[i] / \
            relation_specific_total[i]
        relation_specific_p3[i] = relation_specific_p3[i] / \
            relation_specific_total[i]
        relation_specific_unpruned_p1[i] = relation_specific_unpruned_p1[i] / \
            relation_specific_total[i]
        relation_specific_unpruned_p2[i] = relation_specific_unpruned_p2[i] / \
            relation_specific_total[i]
        relation_specific_unpruned_p3[i] = relation_specific_unpruned_p3[i] / \
            relation_specific_total[i]

    corpus_fileobj.close()
    ret_dict = {}
    ret_dict['macro_pruned_p1'] = pruned_p1
    ret_dict['macro_unpruned_p1'] = unpruned_p1
    ret_dict['micro_pruned_p1'] = sum(
        relation_specific_p1) / len(relation_specific_p1)
    ret_dict['micro_pruned_p2'] = sum(
        relation_specific_p2) / len(relation_specific_p2)
    ret_dict['micro_pruned_p3'] = sum(
        relation_specific_p3) / len(relation_specific_p3)
    ret_dict['micro_unpruned_p1'] = sum(
        relation_specific_unpruned_p1) / len(relation_specific_unpruned_p1)
    ret_dict['micro_unpruned_p2'] = sum(
        relation_specific_unpruned_p2) / len(relation_specific_unpruned_p2)
    ret_dict['micro_unpruned_p3'] = sum(
        relation_specific_unpruned_p3) / len(relation_specific_unpruned_p3)
    ret_dict['sparsity'] = sparsity_dict
    P1_dict = {}
    for i in range(num_relations):
        P1_dict[model.id_to_relation[i]] = relation_specific_p1[i]
    for relation in P1_dict:
        print("{}: {}".format(relation, P1_dict[relation]))
    return ret_dict


def probing(epoch, max_epochs, dataloader, optimizers, lr_schedulers, model: SelfMaskingModel, device):
    total = len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=total)
    avg_loss = .0
    for batch_id, batch in pbar:
        total_loss = .0
        cnt = 0
        for i in range(len(batch[-1])):
            relation_id = batch[-1][i]
            optimizer = optimizers[relation_id]
            scheduler = lr_schedulers[relation_id]
            optimizer.zero_grad()
            input_dict = batch[0][i].to(device)
            labels = batch[1][i].to(device)
            cnt += batch[0][i]['input_ids'].size(0)
            if args.soft_train:
                loss = model.feed_batch(input_dict, labels, relation_id, device)
            else:
                loss = model.feed_batch_straight_through(input_dict, labels, relation_id, device)
            optimizer.step()
            scheduler.step()
            total_loss += loss * batch[0][i]['input_ids'].size(0)
        total_loss /= cnt
        avg_loss += total_loss
        pbar.set_description("Epoch: [{}|{}], Iter: [{}|{}], avg loss: {}".format(
            epoch, max_epochs, batch_id+1, total, total_loss))
    avg_loss /= total
    return avg_loss


def zero_shot_commonsenseQA_eval(model, tokenizer, device):
    """
    Perform zero-shot learning on CommonsenseQA test set
    """
    print("Start Zero-Shot learning on CommonsenseQA test set")
    cnt = 0
    total = 0
    dev_file_pth = "./dev_rand_split.jsonl"
    f = open(dev_file_pth, 'r')
    real_total = 0
    for instance in jsonlines.Reader(f):
        total += 1
        if total <= 610:
            continue
        answerKey = instance['answerKey']
        stem = instance['question']['stem']
        question_concept = instance['question']['question_concept']
        choices = instance['question']['choices']
        candidates = [choice['text'] for choice in choices]
        can2label = {choice['text']: choice['label'] for choice in choices}
        splitted_stem = stem.strip().split()
        _splitted_stem = []
        for i in range(len(splitted_stem)):
            if splitted_stem[i][-1] in {',', '.', '?', '!'}:
                _splitted_stem.append(splitted_stem[i][:-1])
                _splitted_stem.append(splitted_stem[i][-1])
            else:
                _splitted_stem.append(splitted_stem[i])
        splitted_stem = _splitted_stem
        splitted_question_concept = question_concept.strip().split()
        s_id = None
        for i in range(len(splitted_stem)-len(splitted_question_concept)+1):
            match = True
            for j in range(i, i+len(splitted_question_concept)):
                if splitted_stem[j] != splitted_question_concept[j-i]:
                    match = False
                    break
            if match:
                s_id = i
        if s_id is None:
            continue
        real_total += 1
        num_tokens_tobe_masked = len(splitted_question_concept)
        total_ans = {}
        for i in range(s_id, s_id+num_tokens_tobe_masked):
            copyed_stem = deepcopy(splitted_stem)
            obj = copyed_stem[i]
            copyed_stem[i] = tokenizer.mask_token
            prompt = " ".join(copyed_stem)
            ans = zero_shot_csr(model, tokenizer, prompt, obj, candidates, device)
            for can in ans:
                total_ans[can] = total_ans.get(can, 0) + ans[can]
        sorted_candidates = list(sorted(total_ans.keys(), key=total_ans.get, reverse=True))
        pred = can2label[sorted_candidates[0]]
        if pred == answerKey:
            cnt += 1
        print("{}|611".format(total-610))
    print("Acc: {}".format(cnt / real_total))
    return cnt / real_total


def zero_shot_csr(model, tokenizer, prompt, obj, candidates, device):
    """
    prompt: question stem with question concept being masked
    """
    ans = dict()
    for candidate in candidates:
        complete_prompt = prompt + " {} {}".format(tokenizer.sep_token, candidate)
        input_dict = tokenizer(complete_prompt, return_tensors='pt')
        input_ids = input_dict['input_ids']
        mask_index = input_ids[0].tolist().index(tokenizer.mask_token_id)
        outputs = model(**(input_dict).to(device))
        logits = outputs.logits
        can_log_prob = logits[0, mask_index, tokenizer.convert_tokens_to_ids([obj])[0]]
        diff = (can_log_prob).cpu().item()
        ans[candidate] = diff
    return ans


def zero_shot_copa(model, tokenizer, premise, hypothesis, type, device):
    if type == 'effect':
        splitted_premise = premise.strip().split()
        num_tokens_tobe_masked = len(splitted_premise)
        logprobs = []
        for i in range(num_tokens_tobe_masked):
            copyed_premise = deepcopy(splitted_premise)
            obj = copyed_premise[i]
            if obj.lower() in stop_words:
                continue
            copyed_premise[i] = tokenizer.mask_token
            prompt = " ".join(copyed_premise)
            complete_prompt = prompt + " so {}".format(hypothesis)
            input_dict = tokenizer(complete_prompt, return_tensors='pt')
            input_ids = input_dict['input_ids']
            mask_index = input_ids[0].tolist().index(tokenizer.mask_token_id)
            outputs = model(**(input_dict).to(device))
            logits = outputs.logits
            masktoken_log_prob = logits[0, mask_index, tokenizer.convert_tokens_to_ids([obj])[0]].cpu().item()
            logprobs.append(masktoken_log_prob)
        target_premise_score = sum(logprobs)
    else:
        splitted_premise = premise.strip().split()
        num_tokens_tobe_masked = len(splitted_premise)
        logprobs = []
        for i in range(num_tokens_tobe_masked):
            copyed_premise = deepcopy(splitted_premise)
            obj = copyed_premise[i]
            if obj.lower() in stop_words:
                continue
            copyed_premise[i] = tokenizer.mask_token
            prompt = " ".join(copyed_premise)
            complete_prompt = prompt + " because {}".format(hypothesis)
            input_dict = tokenizer(complete_prompt, return_tensors='pt')
            input_ids = input_dict['input_ids']
            mask_index = input_ids[0].tolist().index(tokenizer.mask_token_id)
            outputs = model(**(input_dict).to(device))
            logits = outputs.logits
            masktoken_log_prob = logits[0, mask_index, tokenizer.convert_tokens_to_ids([obj])[0]].cpu().item()
            logprobs.append(masktoken_log_prob)
        target_premise_score = sum(logprobs) / len(logprobs)
    return target_premise_score


def zero_shot_copa_eval(model, tokenizer, device):
    cnt = 0
    total = 0
    for instance in copa_dataset:
        total += 1
        type = instance['question']
        premise = instance['premise']
        choice1 = instance['choice1']
        choice2 = instance['choice2']
        choice1_score = zero_shot_copa(model, tokenizer, premise, choice1, type, device)
        choice2_score = zero_shot_copa(model, tokenizer, premise, choice2, type, device)
        if choice1_score > choice2_score:
            pred = 0
        else:
            pred = 1
        if pred == instance['label']:
            cnt += 1
        print("{}|{}".format(total, len(copa_dataset)))
    # print("Acc: {}".format(cnt / total))
    return cnt / total


def zero_shot_hellaswag_premise(model, tokenizer, premise, hypothesis, device):
    splitted_premise = premise.strip().split()
    num_tokens_tobe_masked = len(splitted_premise)
    logprobs = []
    for i in range(num_tokens_tobe_masked):
        copyed_premise = deepcopy(splitted_premise)
        obj = copyed_premise[i]
        if obj in stop_words:
            continue
        copyed_premise[i] = tokenizer.mask_token
        prompt = " ".join(copyed_premise)
        complete_prompt = prompt + " " + hypothesis
        input_dict = tokenizer(complete_prompt, return_tensors='pt')
        input_ids = input_dict['input_ids']
        mask_index = input_ids[0].tolist().index(tokenizer.mask_token_id)
        outputs = model(**(input_dict).to(device))
        logits = outputs.logits
        masktoken_log_prob = logits[0, mask_index, tokenizer.convert_tokens_to_ids([obj])[0]].cpu().item()
        logprobs.append(masktoken_log_prob)
    target_premise_score = sum(logprobs)
    return target_premise_score


def zero_shot_hellaswag_hypo(model, tokenizer, premise, hypothesis, device):
    splitted_premise = hypothesis.strip().split()
    num_tokens_tobe_masked = len(splitted_premise)
    logprobs = []
    for i in range(num_tokens_tobe_masked):
        copyed_premise = deepcopy(splitted_premise)
        obj = copyed_premise[i]
        if obj in stop_words:
            continue
        copyed_premise[i] = tokenizer.mask_token
        prompt = " ".join(copyed_premise)
        complete_prompt = premise + " " + prompt
        input_dict = tokenizer(complete_prompt, return_tensors='pt')
        input_ids = input_dict['input_ids']
        mask_index = input_ids[0].tolist().index(tokenizer.mask_token_id)
        outputs = model(**(input_dict).to(device))
        logits = outputs.logits
        masktoken_log_prob = logits[0, mask_index, tokenizer.convert_tokens_to_ids([obj])[0]].cpu().item()
        logprobs.append(masktoken_log_prob)
    target_hypothesis_score = sum(logprobs) / len(logprobs)
    return target_hypothesis_score


def zero_shot_hellaswag_eval(model, tokenizer, device, use_premise=True):
    from datasets import load_dataset
    hellaswag_dataset = load_dataset('hellaswag', split='validation')
    cnt = 0
    total = 0
    for instance in tqdm(hellaswag_dataset, total=len(hellaswag_dataset)):
        total += 1
        context = instance['ctx']
        endings = instance['endings']
        scores = []
        for ending in endings:
            if use_premise:
                score = zero_shot_hellaswag_premise(model, tokenizer, context, ending, device)
            else:
                score = zero_shot_hellaswag_hypo(model, tokenizer, context, ending, device)
            scores.append(score)
        pred = torch.argmax(torch.tensor(scores)).cpu().item()
        if str(pred) == instance['label']:
            cnt += 1
    print("Acc: {}".format(cnt / total))
    return cnt / total

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results

def MAS(model, tokenizer, pronoun, candidate_a, candidate_b, sentence_a, sentence_b=None, layer=None, head=None):

    """
    Computes the Maximum Attention Score (MAS) given a sentence, a pronoun and candidates for substitution.
    Parameters
    ----------
    model : transformers.BertModel
        BERT model from BERT visualization that provides access to attention
    tokenizer:  transformers.tokenization.BertTokenizer
        BERT tolenizer
    pronoun: string
        pronoun to be replaced by a candidate
    candidate_a: string
        First pronoun replacement candidate
    candidate_b: string
        Second pronoun replacement candidate
    sentence_a: string
       First, or only sentence
    sentence_b: string (optional)
        Optional, second sentence
    layer: None, int
        If none, MAS will be computed over all layers, otherwise a specific layer
    head: None, int
        If none, MAS will be compputer over all attention heads, otherwise only at specific head
    Returns
    -------

    activity : list
        List of scores [score for candidate_a, score for candidate_b]
    """

    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']

    candidate_a_ids = tokenizer.encode(candidate_a)[1:-1]
    candidate_b_ids = tokenizer.encode(candidate_b)[1:-1]
    pronoun_ids = tokenizer.encode(pronoun)[1:-1]

    if next(model.parameters()).is_cuda:
        attention = model(input_ids.cuda(), token_type_ids=token_type_ids.cuda())[-1]
    else:
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]

    attn = format_attention(attention)

    if next(model.parameters()).is_cuda:
        A = torch.zeros((attn.shape[0], attn.shape[1])).cuda()
        B = torch.zeros((attn.shape[0], attn.shape[1])).cuda()
    else:
        A = torch.zeros((attn.shape[0], attn.shape[1]))
        B = torch.zeros((attn.shape[0], attn.shape[1]))

    if not layer is None:
        assert layer<attn.shape[0], "Maximum layer number "+str(attn.shape[0])+" exceeded"
        layer_slice = slice(layer,layer+1,1)
    else:
        layer_slice = slice(None,None,None)

    if not head is None:
        assert head<attn.shape[1], "Maximum head number "+str(attn.shape[1])+" exceeded"
        head_slice = slice(head,head+1,1)
    else:
        head_slice = slice(None,None,None)

    assert len(find_sub_list(pronoun_ids, input_ids[0].tolist())) > 0, "pronoun not found in sentence"
    assert len(find_sub_list(candidate_a_ids, input_ids[0].tolist())) > 0, "candidate_a not found in sentence"
    assert len(find_sub_list(candidate_b_ids, input_ids[0].tolist())) > 0, "candidate_b not found in sentence"

    for _,src in enumerate(find_sub_list(pronoun_ids, input_ids[0].tolist())):


        for _, tar_a in enumerate(find_sub_list(candidate_a_ids, input_ids[0].tolist())):
            A=A+attn[layer_slice,head_slice, slice(tar_a[0],tar_a[1]+1,1), slice(src[0],src[0]+1,1)].mean(axis=2).mean(axis=2)

        for _, tar_b in enumerate(find_sub_list(candidate_b_ids, input_ids[0].tolist())):
            B=B+attn[layer_slice,head_slice, slice(tar_b[0],tar_b[1]+1,1),slice(src[0],src[0]+1,1)].mean(axis=2).mean(axis=2)
    score = sum((A >= B).flatten()).item()/(A.shape[0]*A.shape[1])
    return [score, 1.0-score]


@torch.no_grad()
def link_prediction_conceptnet100k(model: SelfMaskingModel, tokenizer, device):
    # for samping instances that model almost answer correctly but not in Hits@1
    close_samples = []
    import pickle
    vocab_file = pickle.load(open("./link_prediction_vocab.pkl", 'rb'))
    assert type(vocab_file) == set
    print(len(vocab_file))
    file_path = "./data/CKBC/dev_total.txt"
    pruned_1 = 0
    pruned_2 = 0
    pruned_3 = 0
    pruned_10 = 0
    unpruned_1 = 0
    unpruned_2 = 0
    unpruned_3 = 0
    unpruned_10 = 0
    total = 0
    pruned_mrr = 0
    unpruned_mrr = 0
    relation2sentences = dict()
    relation2sentences['AtLocation'] = [
        "Something you find at [obj] is [subj]."]
    relation2sentences['CapableOf'] = ["[subj] can [obj]."]
    relation2sentences['Causes'] = ["[subj] causes [obj]."]
    relation2sentences['CausesDesire'] = [
        "[subj] would make you want to [obj]."]
    relation2sentences['Desires'] = ["[subj] wants [obj]."]
    relation2sentences['HasA'] = ["[subj] contains [obj]."]
    relation2sentences['HasPrerequisite'] = ["[subj] requires [obj]."]
    relation2sentences['HasProperty'] = ["[subj] can be [obj]."]
    relation2sentences['HasSubevent'] = ["when [subj], [obj]."]
    relation2sentences['IsA'] = ["[subj] is a [obj]."]
    relation2sentences['MadeOf'] = ["[subj] can be made of [obj]."]
    relation2sentences['MotivatedByGoal'] = [
        "you would [subj] because [obj]."]
    relation2sentences['NotDesires'] = ["[subj] does not want [obj]."]
    relation2sentences['PartOf'] = ["[subj] is part of [obj]."]
    relation2sentences['ReceivesAction'] = ["[subj] can be [obj]."]
    relation2sentences['UsedFor'] = ["[subj] may be used for [obj]."]
    lines = open(file_path).readlines()
    for line in tqdm(lines, total=len(lines)):
        relation, head, tail, label = line.strip().split("\t")
        head = head.lower()
        tail = tail.lower()
        if label != "1":
            continue
        if len(head.split(" ")) > 1:
            continue
        if len(tail.split(" ")) > 1:
            continue
        if not relation in relation2sentences:
            continue
        total += 1
        template = relation2sentences[relation][0]
        masked_sentence = template.replace("[subj]", head).replace("[obj]", tokenizer.mask_token)
        relation_id = model.relation_to_id[relation]
        pruning_mask_generator = model.pruning_mask_generators[relation_id]
        hard_samples = []
        for pruning_mask in pruning_mask_generator:
            hard_sample = torch.sigmoid(pruning_mask).to(device)
            hard_sample[hard_sample>0.5] = 1
            hard_sample[hard_sample<=0.5] = 0
            hard_samples.append(hard_sample)
        model.prune(pruning_masks=hard_samples)
        pruned_predictions = utils.LAMA(
            model.pretrained_language_model, tokenizer, device, masked_sentence, topk=5)
        # remove OOV token
        _tmp_pred = []
        for token in pruned_predictions:
            if token in vocab_file:
                _tmp_pred.append(token)
        pruned_predictions = _tmp_pred
        try:
            pos = pruned_predictions.index(tail)
            pruned_mrr += (1 / (pos+1))
            if pos == 0:
                pruned_1 += 1
            if pos in [0, 1]:
                pruned_2 += 1
            if pos in [0, 1, 2]:
                pruned_3 += 1
            if pos <= 9:
                pruned_10 += 1
            close_samples.append([head, relation, tail, masked_sentence, pruned_predictions[:5]])
        except ValueError:
            pass
        model.restore()
        unpruned_predictions = utils.LAMA(
            model.pretrained_language_model, tokenizer, device, masked_sentence.replace('[MASK]', tokenizer.mask_token), topk=5)
        # remove OOV token
        _tmp_pred = []
        for token in unpruned_predictions:
            if token in vocab_file:
                _tmp_pred.append(token)
        unpruned_predictions = _tmp_pred
        try:
            pos = unpruned_predictions.index(tail)
            unpruned_mrr += (1 / (pos+1))
            if pos == 0:
                unpruned_1 += 1
            if pos in [0, 1]:
                unpruned_2 += 1
            if pos in [0, 1, 2]:
                unpruned_3 += 1
            if pos <= 9:
                unpruned_10 += 1
        except ValueError:
            pass
    pruned_p1 = pruned_1 / total
    pruned_p2 = pruned_2 / total
    pruned_p3 = pruned_3 / total
    pruned_p10 = pruned_10 / total
    pruned_mrr = pruned_mrr / total
    unpruned_p1 = unpruned_1 / total
    unpruned_p2 = unpruned_2 / total
    unpruned_p3 = unpruned_3 / total
    unpruned_p10 = unpruned_10 / total
    unpruned_mrr = unpruned_mrr / total
    print("unpruned hits: {}, {}, {}, {}".format(unpruned_p1, unpruned_p2, unpruned_p3, unpruned_p10))
    print("unpruned mrr: {}".format(unpruned_mrr))
    print("pruned hits: {}, {}, {}, {}".format(pruned_p1, pruned_p2, pruned_p3, pruned_p10))
    print("pruned mrr: {}".format(pruned_mrr))
    # if len(close_samples) > 0:
    #     if len(close_samples) > 100:
    #         import random
    #         close_samples = random.sample(close_samples, 100)
    #     with open('sampled_close_triples.json', 'w') as f:
    #         json.dump(close_samples, f)
    #     print("close samples saved!")


def main(args):
    # set random seed
    seed = args.seed
    pl.seed_everything(seed)

    # set computing device
    device = torch.device(args.device)
    logger.info("Using {}".format(device))

    # instantiate dataset and dataloader
    dataset = LAMADataset(args.data_path)
    collator = Collator(dataset.relation_to_id,
                        args.model_name, args.max_length)
    dataloader = DataLoader(dataset, collate_fn=collator,
                            batch_size=args.batch_size, sampler=RandomSampler(dataset))

    # instantiate model
    logger.info("Total number of transformer layers: {}".format(TRANSFORMER_LAYERS.get(args.model_name, -1)))
    logger.info("Bottom Layer Index: {}".format(args.bottom_layer_index))
    logger.info("Top Layer Index: {}".format(args.top_layer_index))
    pl_model = SelfMaskingModel(args.bottom_layer_index, args.top_layer_index,
                                len(dataset.relation_to_id), dataset.relation_to_id, args.model_name, args.lr, init_method=args.init_method)
    pl_model.to(device)
    pl_model.pretrained_language_model.eval()
    freeze_parameters(pl_model.pretrained_language_model)

    # instantiate optimizer and lr scheduler
    optimizers = []
    schedulers = []
    max_steps = len(dataloader) * args.max_epochs
    for ps in pl_model.pruning_mask_generators:
        optimizer = optim.Adam(ps, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, args.warmup*max_steps, max_steps)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    # all_relations = set(pl_model.relation_to_id.keys())
    # all_relations.remove('NotDesires')
    # all_relations.remove('ReceivesAction')
    # triple_combinations = list(combinations(all_relations, 2))
    # for relation in pl_model.relation_to_id:
    #     _id = pl_model.relation_to_id[relation]
    #     if 'mpnet' in args.model_name:
    #         mask_file_pth = "./masks/mpnet-base_{}_36_6_11_init>normal_hard.pickle".format(relation)
    #     else:
    #         mask_file_pth = "./masks/{}_{}_36_6_11_init>normal_hard.pickle".format(args.model_name, relation)
    #     assert os.path.exists(mask_file_pth)
    #     f = open(mask_file_pth, 'rb')
    #     mask = torch.load(f)
    #     pl_model.pruning_mask_generators[_id] = mask
    #     f.close()
    # logger.info("Pruning mask loaded")

    # link prediction
    # link_prediction_conceptnet100k(pl_model, collator.tokenizer, device)
    # exit()

    # best_acc = 0
    # best_rels = []
    # for rels in triple_combinations[::-1]:
    #     # rels = ['Causes', 'Desires', 'MotivatedByGoal']
    #     masks = []
    #     for rel in rels:
    #         id = pl_model.relation_to_id[rel]
    #         mask = pl_model.pruning_mask_generators[id]
    #         _mask = []
    #         for p in mask:
    #             p  = torch.sigmoid(p)
    #             p[p>0.5] = 1
    #             p[p<=0.5] = 0
    #             p = p.bool()
    #             _mask.append(p)
    #         masks.append(_mask)
    #     final_mask = []
    #     for p in zip(*masks):
    #         p = reduce(lambda x, y: torch.logical_or(x, y), p)
    #         final_mask.append(p.float().to(device))
    #     pl_model.prune(final_mask)

        # zero-shot commonsens reasoning
        # prompt = "[MASK] are used during what evening activity?"
        # prompt = "What do people use to cut [MASK]?"
        # prompt = "What [MASK] do you use if you want to combine two words?"
        # prompt = "What would you use to see a [MASK]?"

        # prompt = "When generals want to drop [MASK], what vehicle do they need?"
        # prompt = "What does a [MASK] carpenter use to put holes in objects?"
        # prompt = "Where do students use a [MASK]?"
        # prompt = "What might cause [MASK]?"
        # obj = "utensils"
        # obj = "hair"
        # obj = "preposition"
        # obj = "pen"
        # obj = "clip"
        # obj = "bomb"
        # obj = "thunderstorm"
        # obj = "master"
        # tokenizer = collator.tokenizer
        # model = pl_model.pretrained_language_model
        # orig_input_dict = tokenizer(prompt, return_tensors='pt')
        # mask_index = orig_input_dict['input_ids'][0].tolist().index(tokenizer.mask_token_id)
        # outputs = model(**(orig_input_dict).to(device))
        # logits = outputs.logits
        # log_prob = logits[0, mask_index, tokenizer.convert_tokens_to_ids([obj])[0]]
        # print(log_prob)
        # candidates = ['backpack', 'closet', 'drawer', 'dinner', 'cupboard']
        # candidates = ['pen', 'saw', 'stick', 'scissors', 'keyboard', 'mouse']
        # candidates = ['article', 'adjective', 'conjunction', 'pronoun', 'interjection']
        # candidates = ['office building', 'television show', 'scissors', 'desk drawer', "woman's hair"]
        # candidates = ["friend's house", "paper", "office supply store", "pocket", "classroom"]
        # candidates = ['funny', 'learn new', 'see new', 'play chess', 'surf net']
        # candidates = ['coffee', 'kitchen', 'food store', 'cooking pot', 'cooking']
        # candidates = ['drill', 'learn', 'require obedience', 'understand', 'spoon']
        # ans = zero_shot_csr(model, tokenizer, prompt, obj, candidates, device)
        # print(ans)
        # exit()


        # zero-shot copa evaluation
        # premise = "My body cast a shadow over the grass."
        # choice1 = "The sun was rising."
        # choice2 = "The grass was cut."
        # type = "cause"
        # print(zero_shot_copa(model, tokenizer, premise, choice1, type, device))
        # print(zero_shot_copa(model, tokenizer, premise, choice2, type, device))
        # acc = zero_shot_copa_eval(model, tokenizer, device)
        # acc = zero_shot_commonsenseQA_eval(model, tokenizer, device)
        # acc = zero_shot_hellaswag_eval(model, tokenizer, device, use_premise=True)
    #     print("{} finished".format(rels))
    #     pl_model.restore()
    #     if acc > best_acc:
    #         best_acc = acc
    #         best_rels = rels[:]
    #     print("Current best acc: {} from {}".format(best_acc, best_rels))
    # exit()

    # logger.info("Start validation!")
    # ret_dict = validate(
    #     pl_model, collator.tokenizer, device, args.data_path, len(dataset), args.soft_infer)
    # logger.info("Finish validation!")
    # print("Metrics:")
    # tb = pt.PrettyTable()
    # tb.field_names = ['Model Name', 'Macro-P@1-pruned',
    #                     'Macro-P@1-unpruned', 'Micro-P@1-pruned', 'Micro-P@1-unpruned', 'Micro-P@2-pruned', 'Micro-P@2-unpruned', 'Micro-P@3-pruned', 'Micro-P@3-unpruned', 'Micro-P@5-pruned', 'Micro-P@5-unpruned']
    # tb.add_row([args.model_name, ret_dict['macro_pruned_p1'], ret_dict['macro_unpruned_p1'],
    #             ret_dict['micro_pruned_p1'], ret_dict['micro_unpruned_p1'], ret_dict['micro_pruned_p2'], ret_dict['micro_unpruned_p2'], ret_dict['micro_pruned_p3'], ret_dict['micro_unpruned_p3'], ret_dict['micro_pruned_p5'], ret_dict['micro_unpruned_p5']])
    # print(tb)
    # exit()

    # probe!
    best_macro_pruned_p1 = 0
    best_micro_pruned_p1 = 0
    logger.info(
        "Start to train pruning mask generators using soft approximation")
    logger.info("hard binary mask in training" if not args.soft_train else "soft mask in training")
    logger.info(
        "hard binary mask in inference" if not args.soft_infer else "soft mask in inference")
    best_p1 = 0
    best_p2 = 0
    best_p3 = 0
    for e in range(args.max_epochs):
        loss = probing(e+1, args.max_epochs, dataloader, optimizers,
                       schedulers, pl_model, device)
        logger.info("Epoch {} training finished, loss: {}".format(e+1, loss))
        # validation
        logger.info("Start validation!")
        ret_dict = validate(
            pl_model, collator.tokenizer, device, args.data_path, len(dataset), args.soft_infer)
        logger.info("Finish validation!")
        print("Metrics:")
        tb = pt.PrettyTable()
        tb.field_names = ['Model Name', 'Macro-P@1-pruned',
                          'Macro-P@1-unpruned', 'Micro-P@1-pruned', 'Micro-P@1-unpruned', 'Micro-P@2-pruned', 'Micro-P@2-unpruned', 'Micro-P@3-pruned', 'Micro-P@3-unpruned']
        tb.add_row([args.model_name, ret_dict['macro_pruned_p1'], ret_dict['macro_unpruned_p1'],
                    ret_dict['micro_pruned_p1'], ret_dict['micro_unpruned_p1'], ret_dict['micro_pruned_p2'], ret_dict['micro_unpruned_p2'], ret_dict['micro_pruned_p3'], ret_dict['micro_unpruned_p3']])
        print(tb)
        pprint(ret_dict['sparsity'])
        if ret_dict['macro_pruned_p1'] > best_macro_pruned_p1 or ret_dict['micro_pruned_p1'] > best_micro_pruned_p1:
            if ret_dict['macro_pruned_p1'] > best_macro_pruned_p1:
                best_macro_pruned_p1 = ret_dict['macro_pruned_p1']
            if ret_dict['micro_pruned_p1'] > best_micro_pruned_p1:
                best_micro_pruned_p1 = ret_dict['micro_pruned_p1']
            best_p1 = ret_dict['micro_pruned_p1']
            best_p2 = ret_dict['micro_pruned_p2']
            best_p3 = ret_dict['micro_pruned_p3']
            utils.save_pruning_masks_generators(args,
                                                args.model_name, pl_model.pruning_mask_generators, pl_model.id_to_relation, args.save_dir)
            logger.info(
                "New best pruned P@1 observed, pruning mask generators saved!")
    print("Best pruned P@1: {}".format(best_p1))
    print("Best pruned P@2: {}".format(best_p2))
    print("Best pruned P@3: {}".format(best_p3))


if __name__ == "__main__":
    args = get_args()
    if args.test:
        test_pl(args)
    else:
        main(args)
