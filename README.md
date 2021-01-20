<!--
 * @Author: your name
 * @Date: 2020-10-31 00:05:34
 * @LastEditTime: 2021-01-01 23:44:44
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings EditUse 
 * @FilePath: /LAMA/README.md
-->
# Exploring and Exploiting Latent Commonsense Knowledge in Pretrained Masked Language Models
![](https://img.shields.io/badge/Model%20available-DistilBERT%2FBERT%2FRoBERTa%2FMPNet-blue) ![](https://img.shields.io/badge/paper-pdf-red.svg)


Codebase for the paper "Exploring and Exploiting Latent Commonsense Knowledge in Pretrained Masked Language Models".

**Note**: under maintenance, will be complete soon.

## Current supported models:
- DistilBERT-base
- BERT(base, large, etc.)
- RoBERTa(base, large, etc.)
- MPNet


## Prepare the codebase
```bash
git clone https://github.com/DRSY/LAMP.git && cd LAMP
pip install -r requirements.txt
```

## Run pruning and probing
Specify parameters about probing experiments in a separate **params** file, then run:
```bash
make -f Makefile probe
```
detailed hyperparameters can be found in **probe.sh**.

## Run GLUE
Specify parameters about GLUE experiments in a separate **params** file, then run:
```bash
make -f Makefile glue
```

## Clean the log files
```bash
make -f Makefile clean
```
detailed hyperparameters can be found in **glue.sh**.
