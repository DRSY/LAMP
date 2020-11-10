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
CUDA_VISIBLE_DEVICES=3 python -W ignore -u ./GLUE/main.py --model_name_or_path $model_name --task_name $task_name --max_epochs $max_epochs --gpus $gpus --train_batch_size $train_bs --eval_batch_size $eval_bs | tee "`date`_glue".log