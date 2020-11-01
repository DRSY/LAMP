'''
Author: roy
Date: 2020-10-30 16:29:16
LastEditTime: 2020-11-01 19:36:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /extraction/repo_LAMA.py
'''
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM, BertTokenizer
import copy
import jsonlines
import torch.nn.utils.prune as prune
import pprint
from utils import (Foobar_pruning, device, remove_prune_reparametrization,
                   bernoulli_hard_sampler, bernoulli_soft_sampler, freeze_parameters, restore_init_state, LAMA)
from config import get_args



def main(args):
    model_name = args.model_name
    model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_mask_token = tokenizer.mask_token
    print("Mask token for model {}: {}".format(
        model_name, tokenizer_mask_token))
    cnt = 0
    total = 0
    with open(place_of_birth_path, mode='r', encoding='utf8') as f:
        for sample in tqdm(jsonlines.Reader(f)):
            total += 1
            sentence = sample['masked_sentences'][0]
            obj_label = sample['obj_label']
            if '[MASK]' not in sentence:
                continue
            sentence = sentence.replace('[MASK]', tokenizer_mask_token)
            predicitons = LAMA(model, tokenizer, tokenizer_mask_token,
                               sentence, sentence.replace(tokenizer_mask_token, obj_label))
            if predicitons[0].lower() == obj_label.lower():
                cnt += 1
    mean_p1 = (cnt / total)
    print(mean_p1)


if __name__ == "__main__":
    pprinter = pprint.PrettyPrinter(width=41)
    # pruning mask generator model
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 768))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    inputs = torch.randn(12, 32)
    pruning_mask_logits = model(inputs)
    assert pruning_mask_logits.requires_grad == True
    pruning_mask_probs = torch.sigmoid(pruning_mask_logits)
    soft_samples = bernoulli_soft_sampler(pruning_mask_logits, temperature=0.01)
    hard_samples, log_probs = bernoulli_hard_sampler(pruning_mask_probs)
    assert soft_samples.requires_grad == True, "no grad associated with soft samples"

    # testing
    text = "The capital of England is [MASK]."
    obj_label = "London"
    bert = BertForMaskedLM.from_pretrained('bert-base-cased', return_dict=True)
    bert.eval()
    freeze_parameters(bert)
    init_state = copy.deepcopy(bert.state_dict())
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    predictions = LAMA(bert, tokenizer, text, topk=10)
    pprinter.pprint(predictions)
    parameters_tobe_pruned = [
        (bert.bert.encoder.layer[0].attention.self.query, 'bias')]
    # prune!
    for module, name in parameters_tobe_pruned:
        Foobar_pruning(module, name, soft_samples[0])

    input_dict = tokenizer(text, return_tensors='pt')
    mask_index = input_dict['input_ids'][0].tolist().index(
        tokenizer.mask_token_id)
    labels = input_dict['input_ids'].clone()

    # mask out labels for unwanted tokens(tokens except for [MASK])
    labels.fill_(-100)
    labels[0, mask_index] = tokenizer.convert_tokens_to_ids([obj_label])[0]
    outputs = bert(**input_dict, labels=labels)
    logits = outputs.logits
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    predictions = LAMA(bert, tokenizer, text, topk=10)
    pprinter.pprint(predictions)

    # resrotre initial parameters of BERT
    for module, name in parameters_tobe_pruned:
        remove_prune_reparametrization(module, name)
    restore_init_state(bert, init_state)
    predictions = LAMA(bert, tokenizer, text, topk=10)
    pprinter.pprint(predictions)

    args = get_args()
    main(args)
