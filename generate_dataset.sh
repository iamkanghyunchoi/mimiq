#!/bin/bash

python3 -u hydra_image_gen_ssim_att_map.py --model $1 --num_images $2 --save_prefix $3 --save_path $4 --clamp_training 
