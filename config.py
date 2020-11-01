'''
Author: roy
Date: 2020-11-01 11:16:54
LastEditTime: 2020-11-01 11:17:35
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/config.py
'''
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        "Probing knwoledge in pretrained language model using self-masking")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    args = parser.parse_args()
    return args

