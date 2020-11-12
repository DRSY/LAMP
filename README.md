<!--
 * @Author: your name
 * @Date: 2020-10-31 00:05:34
 * @LastEditTime: 2020-11-12 10:50:27
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings EditUse 
 * @FilePath: /LAMA/README.md
-->
# Weakly Supervised Weights Rescaling and Pruning for Knowledge Mining from Pretrained Language Models

## Test unpruned models(default set ot bert-base-uncased)
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