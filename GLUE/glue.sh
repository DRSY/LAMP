###
# @Author: your name
# @Date: 2020-11-07 15:48:52
 # @LastEditTime: 2020-11-10 09:17:28
 # @LastEditors: Please set LastEditors
# @Description: In User Settings Edit
# @FilePath: /LAMA/GLUE/glue.sh
###
read -p "model name:> " model_name
read -p "task name:> " task_name
read -p "max epochs:> " max_epochs
read -p "gpus:> " gpus
read -p "train_batch_size:> " train_bs
read -p "eval_batch_size:> " eval_bs
read -p "bli:> " bli
read -p "tli:> " tli
read -p "init_method:> " init_method
CUDA_VISIBLE_DEVICES=0 python -W ignore -u ./GLUE/main.py --seed 12 --learning_rate 4e-5 --model_name_or_path $model_name --task_name $task_name --max_epochs $max_epochs --gpus $gpus --train_batch_size $train_bs --eval_batch_size $eval_bs --bli $bli --tli $tli --init_method $init_method --apply_mask | tee "`date`_glue".log