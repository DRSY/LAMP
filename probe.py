'''
Author: roy
Date: 2020-10-31 11:03:02
LastEditTime: 2020-11-10 10:22:58
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/probe.py
'''
from functools import reduce
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import jsonlines
from pprint import pprint
from tqdm import tqdm
import pytorch_lightning as pl
from typing import *
import prettytable as pt
from copy import deepcopy
from nltk.corpus import stopwords
import os
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
def validate(model: SelfMaskingModel, tokenizer, device, corpus_file_path: str, total: int, use_expectation: bool = True):
    """
    validate pruning masks on LAMA dataset
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
        print("{}: {:.3f}".format(relation, P1_dict[relation]))
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
            shuffled_relation_id = (relation_id + 8) % len(model.relation_to_id)
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
    from datasets import load_dataset
    copa_dataset = load_dataset('super_glue', 'copa', split='train')
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
    print("Acc: {}".format(cnt / total))


def zero_shot_hellaswag_premise(model, tokenizer, premise, hypothesis, device):
    splitted_premise = premise.strip().split()
    num_tokens_tobe_masked = len(splitted_premise)
    logprobs = []
    for i in range(num_tokens_tobe_masked):
        copyed_premise = deepcopy(splitted_premise)
        obj = copyed_premise[i]
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
    
    for relation in pl_model.relation_to_id:
        _id = pl_model.relation_to_id[relation]
        mask_file_pth = "./masks/{}_{}_24_8_11_init>normal.pickle".format(args.model_name, relation)
        assert os.path.exists(mask_file_pth)
        f = open(mask_file_pth, 'rb')
        mask = torch.load(f)
        pl_model.pruning_mask_generators[_id] = mask
        f.close()
    logger.info("Pruning mask loaded")
    # rels = ['AtLocation']
    # masks = []
    # for rel in rels:
    #     id = pl_model.relation_to_id[rel]
    #     mask = pl_model.pruning_mask_generators[id]
    #     _mask = []
    #     for p in mask:
    #         p  = torch.sigmoid(p)
    #         p[p>0.5] = 1
    #         p[p<=0.5] = 0
    #         p = p.bool()
    #         _mask.append(p)
    #     masks.append(_mask)
    # final_mask = []
    # for p in zip(*masks):
    #     p = reduce(lambda x, y: torch.logical_or(x, y), p)
    #     final_mask.append(p.float().to(device))
    # pl_model.prune(final_mask)

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
    tokenizer = collator.tokenizer
    model = pl_model.pretrained_language_model
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
    # zero_shot_copa_eval(model, tokenizer, device)
    # zero_shot_commonsenseQA_eval(model, tokenizer, device)
    zero_shot_hellaswag_eval(model, tokenizer, device, use_premise=False)
    exit()


    logger.info("Start validation!")
    ret_dict = validate_mismatched(
        pl_model, collator.tokenizer, device, args.data_path, len(dataset), args.soft_infer)
    logger.info("Finish validation!")
    print("Metrics:")
    tb = pt.PrettyTable()
    tb.field_names = ['Model Name', 'Macro-P@1-pruned',
                        'Macro-P@1-unpruned', 'Micro-P@1-pruned', 'Micro-P@1-unpruned', 'Micro-P@2-pruned', 'Micro-P@2-unpruned', 'Micro-P@3-pruned', 'Micro-P@3-unpruned']
    tb.add_row([args.model_name, ret_dict['macro_pruned_p1'], ret_dict['macro_unpruned_p1'],
                ret_dict['micro_pruned_p1'], ret_dict['micro_unpruned_p1'], ret_dict['micro_pruned_p2'], ret_dict['micro_unpruned_p2'], ret_dict['micro_pruned_p3'], ret_dict['micro_unpruned_p3']])
    print(tb)
    exit()

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
        # validation
        logger.info("Epoch {} training finished, loss: {}".format(e+1, loss))
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


# results for unpruned models on ConceptNet subset of LAMA

# roberta-large: 18.56 , loss: 43.65
# albert-xxlarge-v2: 16.44, loss: 16.65
# roberta-base: 15.51, loss: 18.45
# bert-large-uncased: 15.13, loss: 6.59
# bert-large-cased: 15.06, loss: 6.65
# albert-xlarge-v2: 13.97, loss: 15.52
# bert-base-uncased: 12.84, loss: 6.80
# distilroberta-base: 12.49, loss: 18.47
# bert-base-cased: 12.04, loss: 6.97
# albert-large-v2: 11.66, loss: 14.56
# distilbert-base-uncased: 11.37, loss: 6.77
# distilbert-base-cased: 9.82, loss: 7.02
# albert-base-v2: 7.66,loss: 14.78


# results for pruned models
# bert-base-cased:
# bert-base-uncased:
# bert-large-cased:
# bert-large-uncased:
# roberta-base:
# roberta-large:
# distilroberta-base
# distilbert-base-cased:
# distilbert-base-uncased: