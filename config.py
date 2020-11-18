'''
Author: roy
Date: 2020-11-01 11:16:54
LastEditTime: 2020-11-09 09:20:19
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

# allowed models
MODEL_NAMES = ['bert-base-cased', 'bert-base-uncased', 'distilbert-base-cased', 'distilbert-base-uncased', 'distilroberta-base', 'funnel-transformer/large-base', 'funnel-transformer/medium-base', 'bert-base-cased-finetuned-mrpc', 'SpanBERT/spanbert-base-cased', 'SpanBERT/spanbert-large-cased',
               'bert-large-uncased', 'bert-large-cased', 'roberta-base', 'roberta-large', 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'dbmdz/bert-base-cased-finetuned-conll03-english']

# layers
TRANSFORMER_LAYERS = {
    'funnel-transformer/medium-base': 12,
    'funnel-transformer/large-base': 24,
    'bert-base-uncased': 12,
    'bert-base-cased': 12,
    'SpanBERT/spanbert-base-cased': 12,
    'SpanBERT/spanbert-large-cased': 24,
    'bert-large-uncased': 24,
    'bert-large-cased': 24,
    'distilbert-base-uncased': 6,
    'distilbert-base-cased': 6,
    'roberta-base': 12,
    'roberta-large': 24,
}


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
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', required=True, choices=MODEL_NAMES,
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
                        default=False, help="whether use soft mask or hard binary mask during inference")
    parser.add_argument('--soft_train', action='store_true', default=False,
                        help="whether to use re-parametrization during training")
    parser.add_argument('--bottom_layer_index', type=int, default=0)
    parser.add_argument('--top_layer_index', type=int, default=11)
    parser.add_argument('--init_method', type=str, default='uniform', choices=[
                        'uniform', 'normal', 'ones', 'zeros', '2.95', '2.75', '1.38', '0.85', '0.62', '0.41'], help="initialization method for pruning mask generators which determine the initial sparsity of pruning masks")
    parser.add_argument('--l0', default=False, action='store_true',
                        help="whether add l0 penalty to pruning masks")
    args = parser.parse_args()
    if args.model_name not in MODEL_NAMES:
        raise Exception("model name {} not in predefined list: {}".format_map(
            args.model_name, MODEL_NAMES))
    pprint(vars(args))
    return args


if __name__ == "__main__":
    get_args()
