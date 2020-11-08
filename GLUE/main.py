'''
Author: roy
Date: 2020-11-07 15:49:03
LastEditTime: 2020-11-08 11:32:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/GLUE/main.py
'''
import sys
import os
sys.path.append(os.getcwd())

from GLUE.glue_model import *
from GLUE.glue_datamodule import *


def parse_args(args=None):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GLUEDataModule.add_argparse_args(parser)
    parser = GLUETransformer.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)


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
    data_module, pl_model, trainer = main(args)
    trainer.fit(pl_model, data_module)
