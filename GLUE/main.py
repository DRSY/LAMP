'''
Author: roy
Date: 2020-11-07 15:49:03
LastEditTime: 2020-11-08 19:33:50
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/GLUE/main.py
'''
import sys
import os
sys.path.append(os.getcwd())

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:  %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)

from GLUE.glue_model import *
from GLUE.glue_datamodule import *
from torch.nn.utils import prune


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
    assert len(masks) == len(parameters_tobe_pruned), f"{parameters_tobe_pruned} != {len(masks)}"
    for mask, (module, name) in zip(masks, parameters_tobe_pruned):
        prune.custom_from_mask(module, name, mask)
    logger.info("Pre-computed mask applied to {}".format(model_name))
    
    


def parse_args():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GLUEDataModule.add_argparse_args(parser)
    parser = GLUETransformer.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=42)
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
    trainer.fit(pl_model, data_module)
