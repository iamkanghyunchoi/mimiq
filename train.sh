#!/bin/bash

python3 -u main.py --conf_path $1 --id $2 --lrs $3 --qw $4 --qa $5 --head_dist_coef $6 --dataset_path $7 --lr_policy $8 --lr_step $9 --aq_mode ${10} --bs 16
