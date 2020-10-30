'''
Author: roy
Date: 2020-10-30 16:29:16
LastEditTime: 2020-10-30 22:43:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /extraction/repo_LAMA.py
'''
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import copy
import jsonlines
import torch.nn.utils.prune as prune
from utils import Foobar_pruning, device, remove_prune_reparametrization

conceptNet_path = "./test.jsonl"
place_of_birth_path = "./place_of_birth_test.jsonl"


def LAMA(model, tokenizer, mask_token, input_w_mask, input_wo_mask):
    # print(input_w_mask)
    inputs = tokenizer(input_w_mask, return_tensors='pt')
    # print(inputs['input_ids'][0].tolist())
    mask_id = inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)
    inputs.to(device)
    labels = tokenizer(input_wo_mask, return_tensors='pt')['input_ids']
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits[0, mask_id], dim=-1)
    _, indices = torch.topk(probs, k=5)
    predictions = []
    for token in tokenizer.decode(indices).split(" "):
        predictions.append(token)
    return predictions


def main(args):
    model_name = args.model_name
    model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
    model.to(device)
    init_state_dict = copy.deepcopy(model.state_dict())
    # parameters to be pruned
    parameters_to_prune = [
        (model.bert.encoder.layer[0].attention.self.query, 'bias')]
    # prune!
    for module, name in parameters_to_prune:
        Foobar_pruning(module, name)

    # make the pruning permenant
    for module, name in parameters_to_prune:
        remove_prune_reparametrization(module, name)

    # use the pruned model in LAMA probe

    # restore the initial weights
    model.load_state_dict(init_state_dict)
    exit()
    # prune.global_unstructured(parameters_to_prune, prune.L1Unstructured, amount=0.1)
    print(model.bert.encoder.layer[0].attention.self.query.weight)
    print(model.bert.encoder.layer[0].attention.self.query.weight_orig)
    print(
        model.bert.encoder.layer[0].attention.self.query.weight_mask.eq(0).sum())
    print(
        model.bert.encoder.layer[0].attention.self.query.weight_mask.nelement())
    exit()
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
    parser = argparse.ArgumentParser(
        "Probing knwoledge in pretrained language model using self-masking")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    args = parser.parse_args()
    main(args)
