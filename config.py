'''
Author: roy
Date: 2020-11-01 11:16:54
LastEditTime: 2020-11-03 16:40:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/config.py
'''
import argparse
import logging
from pytorch_lightning import Trainer
from pprint import pprint

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:  %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)

conceptNet_path = "./data/ConceptNet/test.jsonl"
place_of_birth_path = "./data/Google_RE/place_of_birth_test.jsonl"
place_of_death_path = "./data/Google_RE/place_of_death_test.jsonl"

MODEL_NAMES = ['bert-base-cased', 'bert-base-uncased',
               'bert-large-uncased', 'bert-large-cased', 'roberta-base', 'roberta-large']


def get_args():
    parser = argparse.ArgumentParser(
        "Probing knwoledge in pretrained language model using self-masking")
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--temperature', type=float, default=0.1,
                        help="temperature of bernoulli re-parameterization tricks")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size", required=True)
    parser.add_argument('--data_path', type=str, default='./data/ConceptNet/test.jsonl',
                        help="path of knowledge source, ends with .jsonl")
    parser.add_argument('--save_dir', type=str, default='./masks',
                        help="directory to save trained pruning mask generators")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', required=True,
                        help="name of pretrained language model")
    parser.add_argument('--max_length', type=int, default=20,
                        help="max length of input cloze question")
    parser.add_argument('--lr', type=float, default=2e-4, help="learning rate")
    parser.add_argument('--warmup', type=float, default=0.1,
                        help="ration of total max steps of warm up stage")
    parser.add_argument('--device', type=str, default='cuda:3', help="gpu id")
    parser.add_argument('--test', action='store_true', default=False,
                        help="whether trigger test utility rather than official training")
    parser.add_argument('--soft_infer', action='store_true',
                        default=False, help="")
    parser.add_argument('--bottom_layer_index', type=int, default=0)
    parser.add_argument('--top_layer_index', type=int, default=11)
    args = parser.parse_args()
    pprint(vars(args))
    return args


if __name__ == "__main__":
    get_args()
