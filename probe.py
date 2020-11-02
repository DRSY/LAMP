'''
Author: roy
Date: 2020-10-31 11:03:02
LastEditTime: 2020-11-03 00:08:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/probe.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.serialization import save
from transformers import get_linear_schedule_with_warmup
import jsonlines
from pprint import pprint
from tqdm import tqdm
import pytorch_lightning as pl

from config import logger, get_args
from model import SelfMaskingModel
from data import LAMADataset, Collator, DataLoader, Dataset, RandomSampler
import utils


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
    pl_model = SelfMaskingModel(
        len(dataset.relation_to_id), dataset.relation_to_id, args.model_name, args.lr)
    pl_model.to(device)

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
    lowest_loss = 1e9
    logger.info(
        "Start to train pruning mask generators using soft approximation")
    for e in range(args.max_epochs):
        loss = probing(e+1, args.max_epochs, dataloader, optimizers,
                       schedulers, pl_model, device)
        if loss < lowest_loss:
            lowest_loss = loss
            utils.save_pruning_masks_generators(
                args.model_name, pl_model.pruning_mask_generators, pl_model.id_to_relation, args.save_dir)
            logger.info(
                "Updateed lowest loss: {}, pruning mask generators saved!")
        logger.info("Epoch {} finished".format(e+1))


if __name__ == "__main__":
    args = get_args()
    if args.test:
        test_pl(args)
    else:
        main(args)
