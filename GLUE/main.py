'''
Author: roy
Date: 2020-11-07 15:49:03
LastEditTime: 2020-11-08 14:48:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/GLUE/main.py
'''
import sys
import os
sys.path.append(os.getcwd())

from GLUE.glue_model import *
from GLUE.glue_datamodule import *


def load_masks(pl_model: GLUETransformer, model_name, bli, tli):
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
    for ii in range(bli, tli+1):
        pass


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
