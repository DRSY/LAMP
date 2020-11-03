'''
Author: roy
Date: 2020-10-31 11:03:02
LastEditTime: 2020-11-04 00:08:18
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/probe.py
'''
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import jsonlines
from pprint import pprint
from tqdm import tqdm
import pytorch_lightning as pl
from typing import *
import prettytable as pt

from config import logger, get_args
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


def validate(model: SelfMaskingModel, tokenizer, device, corpus_file_path: str, total: int, use_expectation: bool = True):
    """
    validate pruning masks on LAMA dataset
    """
    corpus_fileobj = open(corpus_file_path, mode='r', encoding='utf-8')
    num_relations = len(model.pruning_mask_generators)
    relation_specific_p1 = [.0] * num_relations
    relation_specific_total = [0] * num_relations
    relation_specific_unpruned_p1 = [.0] * num_relations
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
                    hard_sample = utils.bernoulli_hard_sampler(
                        torch.sigmoid(pruning_mask), require_logprob=False).to(device)  # use discrete Bernoulli variables
                hard_samples.append(hard_sample)
            model.prune(pruning_masks=hard_samples)
            pruned_predictions = utils.LAMA(
                model.pretrained_language_model, tokenizer, device, masked_sentence, topk=5)
            try:
                pos = pruned_predictions.index(obj_label)
                if pos == 0:
                    pruned_top1 += 1
            except ValueError:
                pass
            model.restore()
            unpruned_predictions = utils.LAMA(
                model.pretrained_language_model, tokenizer, device, masked_sentence, topk=5)
            try:
                pos = unpruned_predictions.index(obj_label)
                if pos == 0:
                    unpruned_top1 += 1
            except ValueError:
                pass
        else:
            masked_sentences = instance['masked_sentences']
            # masked_sentences = ['The capital of England is [MASK].']
            obj_label = instance['obj_label'].lower()
            # obj_label = 'London'.lower()
            relation = instance['pred']
            relation_id = model.relation_to_id[relation]
            relation_specific_total[relation_id] += 1
            # pruning_mask_generator = model.pruning_mask_generators[relation_id]
            # hard_samples = []
            # for pruning_mask in pruning_mask_generator:
            #     if use_expectation:
            #         hard_sample = torch.sigmoid(pruning_mask).to(
            #             device)  # use the expectation values
            #     else:
            #         hard_sample = utils.bernoulli_hard_sampler(
            #             torch.sigmoid(pruning_mask), require_logprob=False).to(device)  # use discrete Bernoulli variables
            #     hard_samples.append(hard_sample)
            # model.prune(pruning_masks=hard_samples)
            # pruned_predictions = utils.LAMA(
            #     model.pretrained_language_model, tokenizer, device, masked_sentences[0], topk=5)
            # try:
            #     pos = pruned_predictions.index(obj_label)
            #     if pos == 0:
            #         pruned_top1 += 1
            #         relation_specific_p1[relation_id] += 1
            # except ValueError:
            #     pass
            # model.restore()
            unpruned_predictions = utils.LAMA(
                model.pretrained_language_model, tokenizer, device, masked_sentences[0].replace('[MASK]', tokenizer.mask_token), topk=5)
            # print(pruned_predictions)
            # print(unpruned_predictions)
            try:
                pos = unpruned_predictions.index(obj_label)
                if pos == 0:
                    unpruned_top1 += 1
                    relation_specific_unpruned_p1[relation_id] += 1
            except ValueError:
                pass
    # macro-average
    pruned_p1 = pruned_top1 / total
    unpruned_p1 = unpruned_top1 / total

    # micro-average
    for i in range(num_relations):
        relation_specific_p1[i] = relation_specific_p1[i] / \
            relation_specific_total[i]
        relation_specific_unpruned_p1[i] = relation_specific_unpruned_p1[i] / \
            relation_specific_total[i]

    corpus_fileobj.close()
    ret_dict = {}
    ret_dict['macro_pruned_p1'] = pruned_p1
    ret_dict['macro_unpruned_p1'] = unpruned_p1
    ret_dict['micro_pruned_p1'] = sum(
        relation_specific_p1) / len(relation_specific_p1)
    ret_dict['micro_unpruned_p1'] = sum(
        relation_specific_unpruned_p1) / len(relation_specific_unpruned_p1)
    return ret_dict


def probing(epoch, max_epochs, dataloader, optimizers, lr_schedulers, model, device):
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
            loss = model.feed_batch(input_dict, labels, relation_id, device)
            optimizer.step()
            scheduler.step()
            total_loss += loss
        total_loss /= cnt
        avg_loss += total_loss
        pbar.set_description("Epoch: [{}|{}], Iter: [{}|{}], total loss: {}".format(
            epoch, max_epochs, batch_id+1, total, total_loss))
    avg_loss /= total
    return avg_loss


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
    logger.info("Bottom Layer Index: {}".format(args.bottom_layer_index))
    logger.info("Top Layer Index: {}".format(args.top_layer_index))
    pl_model = SelfMaskingModel(args.bottom_layer_index, args.top_layer_index,
                                len(dataset.relation_to_id), dataset.relation_to_id, args.model_name, args.lr)
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

    # probe!
    best_macro_pruned_p1 = 0
    best_micro_pruned_p1 = 0
    logger.info(
        "Start to train pruning mask generators using soft approximation")
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
                          'Macro-P@1-unpruned', 'Micro-P@1-pruned', 'Micro-P@1-unpruned']
        tb.add_row([args.model_name, ret_dict['macro_pruned_p1'], ret_dict['macro_unpruned_p1'],
                    ret_dict['micro_pruned_p1'], ret_dict['micro_unpruned_p1']])
        print(tb)
        if ret_dict['macro_pruned_p1'] > best_macro_pruned_p1 or ret_dict['micro_pruned_p1'] > best_micro_pruned_p1:
            if ret_dict['macro_pruned_p1'] > best_macro_pruned_p1:
                best_macro_pruned_p1 = ret_dict['macro_pruned_p1']
            if ret_dict['micro_pruned_p1'] > best_micro_pruned_p1:
                best_micro_pruned_p1 = ret_dict['micro_pruned_p1']
            utils.save_pruning_masks_generators(
                args.model_name, pl_model.pruning_mask_generators, pl_model.id_to_relation, args.save_dir)
            logger.info(
                "New best pruned P@1 observed, pruning mask generators saved!")


if __name__ == "__main__":
    args = get_args()
    if args.test:
        test_pl(args)
    else:
        main(args)


# results for unpruned models
# roberta-large: 18.56
# roberta-base: 15.51
# bert-large-uncased: 15.13
# bert-large-cased: 15.06
# bert-base-uncased: 12.87
# distilroberta-base: 12.49
# bert-base-cased: 12.04
# distilbert-base-uncased: 11.37
# distilbert-base-cased: 9.82


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