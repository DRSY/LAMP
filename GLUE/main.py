'''
Author: roy
Date: 2020-11-07 15:49:03
LastEditTime: 2020-11-12 11:58:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/GLUE/main.py
'''
import os
import sys
sys.path.append(os.getcwd())

from torch.nn.utils import prune
from GLUE.glue_datamodule import *
from GLUE.glue_model import *
import logging
from typing import *

from pprint import pprint

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:  %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


def load_masks(model_name: str, bli: int, tli: int, relations: List, init_method: str):
    masks = []
    for relation in relations:
        mask_pth = "/home/roy/commonsense/LAMA/masks/{}_{}_{}_{}_{}_init>{}.pickle".format(model_name, relation, (tli-bli+1)*6, bli, tli, init_method)
        with open(mask_pth, mode='rb') as f:
            mask = torch.load(f)
            masks.append(mask)
    return masks


def union_masks(*masks):
    thresholded_masks = []
    for mask in masks:
        tmp = []
        assert isinstance(mask[0], torch.nn.Parameter)
        for matrix in mask:
            prob = torch.sigmoid(matrix.data)
            prob[prob > 0.5] = 1
            prob[prob <= 0.5] = 0
            prob = prob.bool()
            tmp.append(prob)
        thresholded_masks.append(tmp)
    final_masks = []
    for mask_for_all_relations in zip(thresholded_masks):
        tmp_mask = torch.logical_or(*mask_for_all_relations)
        final_masks.append(tmp_mask)
    return final_masks


def apply_masks(pl_model: GLUETransformer, model_name, bli, tli, masks):
    backbone = pl_model.model
    model_type = model_name.split('-')[0]
    assert hasattr(backbone, model_type), f"LM does not have {model_type}"
    if 'roberta' in model_name:
        layers = backbone.roberta.encoder.layer
    elif 'distil' in model_name:
        layers = backbone.distilbert.transformer.layer
    else:
        layers = backbone.bert.encoder.layer

    # load pre-trained masks
    parameters_tobe_pruned = []
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
    assert len(masks) == len(
        parameters_tobe_pruned), f"{parameters_tobe_pruned} != {len(masks)}"
    for mask, (module, name) in zip(masks, parameters_tobe_pruned):
        prune.custom_from_mask(module, name, mask)
    logger.info("Pre-computed mask applied to {}".format(model_name))


def parse_args():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GLUEDataModule.add_argparse_args(parser)
    parser = GLUETransformer.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--apply_mask', default=False, action='store_true', help="whether to apply pre-computed mask")
    parser.add_argument('--bli', type=int, default=None)
    parser.add_argument('--tli', type=int, default=None)
    parser.add_argument('--relations', nargs='+', type=str)
    parser.add_argument('--init_method', type=str)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed)
    dm = GLUEDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup('fit')
    model = GLUETransformer(num_labels=dm.num_labels,
                            eval_splits=dm.eval_splits, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    return dm, model, trainer


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    data_module, pl_model, trainer = main(args)
    if args.apply_mask and args.bli is not None and args.tli is not None and args.init_method is not None:
        # load masks
        logger.info("Loading pre-computed masks")
        masks = load_masks(args.model_name, args.bli, args.tli, args.relations, args.init_method)
        # union masks
        logger.info("Unifying masks")
        final_mask = union_masks(*masks)
        # apply masks
        logger.info("Applying final unioned mask")
        apply_masks(pl_model, args.model_name, args.bli, args. tli, final_mask)
    logger.info("Start training on GLUE")
    trainer.fit(pl_model, data_module)
