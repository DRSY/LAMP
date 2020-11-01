'''
Author: roy
Date: 2020-11-01 11:08:20
LastEditTime: 2020-11-01 16:35:51
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/data.py
'''
from utils import LAMA
import torch
import jsonlines
from torch.utils.data import DataLoader, Dataset, RandomSampler
from typing import *
from transformers import AutoTokenizer

from pprint import pprint
from config import (conceptNet_path, place_of_birth_path,
                    place_of_death_path, logger, get_args)


class LAMADataset(Dataset):
    """
    Customized Dataset for loading LAMA dataset
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.datas = []
        self.relation_to_id = dict()
        self.read_data(self.path)

    def read_data(self, path: str):
        assert path.endswith('jsonl'), "not a jsonline file"
        logger.info("start reading file {}".format(path))
        with open(path, mode='r', encoding='utf-8') as f:
            for instance in jsonlines.Reader(f):
                masked_sentences = instance['masked_sentences']
                relation = instance['pred']
                if not relation in self.relation_to_id:
                    self.relation_to_id[relation] = len(self.relation_to_id)
                obj_label = instance['obj_label']
                if '[MASK]' not in masked_sentences[0]:
                    continue
                self.datas.append((masked_sentences[0], obj_label, relation))
        logger.info("finish reading file {}, get {} instances".format(
            path, len(self.datas)))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index: int):
        return self.datas[index]


class Collator(object):
    """
    Collator class for gathering samples within a mini-batch
    """

    def __init__(self, relation_to_id: dict, model_name: str, max_length: int) -> None:
        self.relation2id = relation_to_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.max_length = max_length
        logger.info("Colaltor initialized")

    def get_label(self, input_ids: List[int], obj_label: str):
        mask_token_index = input_ids.index(self.mask_token_id)
        labels = input_ids.clone()
        labels.fill_(-100)
        labels[mask_token_index] = self.tokenizer.convert_tokens_to_ids([obj_label])[
            0]
        return labels

    def __call__(self, data_batch: List):
        def merge_batch(data_batch: List):
            masked_sentences = [data[0] for data in data_batch]
            obj_labels = [data[1] for data in data_batch]
            relations = [data[-1] for data in data_batch]
            return masked_sentences, obj_labels, relations

        if self.tokenizer.mask_token != '[MASK]':
            for data in data_batch:
                data[0] = data[0].replace('[MASK]', self.mask_token)
        bs = len(data_batch)
        masked_sentences, obj_labels, relations = merge_batch(
            data_batch=data_batch)
        relations_in_batch = set()
        tmp_batch_dict = dict()
        for i in range(bs):
            relation_id = self.relation2id.get(relations[i])
            relations_in_batch.add(relation_id)
            if not relation_id in tmp_batch_dict:
                tmp_batch_dict[relation_id] = {
                    'masked_sentences': [], 'obj_labels': []}
            else:
                tmp_batch_dict[relation_id]['masked_sentences'].append(
                    masked_sentences[i])
                tmp_batch_dict[relation_id]['obj_labels'].append(obj_labels[i])
        relations_in_batch = list(relations_in_batch)
        num_relations = len(relations_in_batch)
        input_dict_list = []
        labels_list = []
        for relation_id in relations_in_batch:
            batch_input_dict = self.tokenizer(tmp_batch_dict[relation_id]['masked_sentences'], max_length=self.max_length,
                                              padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True)
            batch_input_ids = batch_input_dict['input_ids'].tolist()
            labels = []
            for i in range(len(tmp_batch_dict[relation_id]['masked_sentences'])):
                label = self.get_label(
                    batch_input_ids[i], tmp_batch_dict[relation_id]['obj_labels'][i])
                labels.append(label)
            labels = torch.tensor(labels).type(
                batch_input_dict['input_ids'].dtype)
            tmp_batch_dict[relation_id]['labels'] = labels
            tmp_batch_dict[relation_id]['input_dict'] = batch_input_dict
            input_dict_list.append(batch_input_dict)
            labels_list.append(labels)
        return input_dict_list, labels_list, relations_in_batch


def test():
    args = get_args()

    toy_dataset = LAMADataset(conceptNet_path)
    relation_to_id = toy_dataset.relation_to_id
    pprint(relation_to_id)
    collator = Collator(relation_to_id, args.model_name, args.max_length)


if __name__ == "__main__":
    test()
