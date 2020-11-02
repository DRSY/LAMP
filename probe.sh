###
# @Author: roy
# @Date: 2020-11-02 16:54:54
 # @LastEditTime: 2020-11-02 18:46:11
 # @LastEditors: Please set LastEditors
# @Description: In User Settings Edit
# @FilePath: /LAMA/probe.sh
###
read -p "model name:> " model_name
read -p "warmup:> " warmup
read -p "device:> " device
read -p "batch size:> " bs
read -p "max epochs:> " max_epochs
read -p "temperature:> " temperature
python -W ignore -u probe.py --model_name $model_name --lr 2e-5 --seed 100 --warmup $warmup --max_epochs $max_epochs --device cuda:$device --batch_size $bs --temperature $temperature | tee train.log
