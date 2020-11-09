###
# @Author: roy
# @Date: 2020-11-02 16:54:54
 # @LastEditTime: 2020-11-09 22:56:29
 # @LastEditors: Please set LastEditors
# @Description: In User Settings Edit
# @FilePath: /LAMA/probe.sh
###
read -p "model name:> " model_name
read -p "warmup:> " warmup
read -p "device:> " device
read -p "lr:> " lr
read -p "batch size:> " bs
read -p "max epochs:> " max_epochs
read -p "temperature:> " temperature
read -p "bottom layer index(0-11):> " bli
read -p "top layer index(0-11):> " tli
read -p "init method:> " init_method
python -W ignore -u probe.py --model_name $model_name --lr $lr --seed 100 --warmup $warmup --max_epochs $max_epochs --device cuda:$device --batch_size $bs --temperature $temperature --soft_infer --soft_train --bottom_layer_index $bli --top_layer_index $tli --init_method $init_method | tee "`date`_probe".log