###
# @Author: your name
# @Date: 2020-11-07 15:48:52
# @LastEditTime: 2020-11-08 11:31:49
# @LastEditors: Please set LastEditors
# @Description: In User Settings Edit
# @FilePath: /LAMA/GLUE/glue.sh
###
read -p "model name:> " model_name
read -p "task name:> " task_name
read -p "max epochs:> " max_epochs
read -p "gpus:> " gpus
python -W ignore -u main.py --model_name_or_path $model_name --task_name $task_name --max_epochs $max_epochs --gpus $gpus
