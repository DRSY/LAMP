<!--
 * @Author: your name
 * @Date: 2020-10-31 00:05:34
 * @LastEditTime: 2020-11-18 20:33:40
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings EditUse 
 * @FilePath: /LAMA/README.md
-->
# Exploring and Exploiting Latent Commonsense Knowledge in Pretrained Masked Language Models
![](https://img.shields.io/badge/Language%20Model%20Pruning(LAMP)-DistilBERT%2FBERT%2FMPNet-blue.svg)
Codebase for the paper "Exploring and Exploiting Latent Commonsense Knowledge in Pretrained Masked Language Models".

## Test unpruned models(default set to bert-base-uncased)
```bash
make -f Makefile test
```

## Run probing
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
