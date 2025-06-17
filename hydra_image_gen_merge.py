import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import requests

# option file should be modified according to your expriment
from options import Option

from dataloader import DataLoader
from trainer import Trainer

import utils as utils
from quant_utils.quant_modules import *

import timm

from torch.utils.data import TensorDataset

from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention

# for model in ["vit_tiny_patch16_224", "deit_tiny_patch16_224","vit_small_patch16_224", "deit_small_patch16_224","vit_base_patch16_224", "deit_base_patch16_224"]:
# for model in ["swin_tiny_patch4_window7_224","swin_small_patch4_window7_224","swin_base_patch4_window7_224"]:

for ssim_coef in [1.0]:
    for class_coef in [1.0]:
        for tv_coef in [2.5e-5]:
                
            SAVE_PATH = sys.argv[1] # SAVE PATH
            model = sys.argv[2]
                
            model_origin = timm.create_model(model, pretrained=False).eval()
            IMG_SIZE = model_origin.patch_embed.img_size[0]
            PATCH_SIZE = model_origin.patch_embed.patch_size[0]
            NUM_PATCHES = model_origin.patch_embed.num_patches
            if 'swin' in model:
                NUM_HEADS = []
                NUM_WINDOWS = []
                for m in model_origin.modules():
                    if isinstance(m, WindowAttention):
                        NUM_HEADS.append(m.num_heads)
                        NUM_WINDOWS.append((model_origin.num_features // m.dim) ** 2)
                        WINDOW_AREA = m.window_area
                MODEL_DEPTH = len(NUM_HEADS)
            else:
                NUM_HEADS = model_origin.blocks[0].attn.num_heads
                MODEL_DEPTH = len(model_origin.blocks)
            
            PATH = os.path.join(SAVE_PATH, f"{model}_ssim_{ssim_coef}_class_{class_coef}_tv_{tv_coef}/")
            image_name_list = []
            class_label_name_list = []

            for filename in os.listdir(PATH):
                if "images.pt" in filename:
                    image_name_list.append(filename)
                elif "class_labels.pt" in filename:
                    class_label_name_list.append(filename)
                    
            image_name_list.sort(key=lambda name: int(name.split("_")[0]))
            class_label_name_list.sort(key=lambda name: int(name.split("_")[0]))
            
            image_list = []
            class_list = []

            for image_name, class_name in zip(image_name_list, class_label_name_list):
                print(image_name)
                image = torch.load(os.path.join(PATH,image_name))
                class_label = torch.load(os.path.join(PATH, class_name))

                image_list.append(image)
                class_list.append(class_label)
                
                
            images = torch.cat(image_list)
            class_labels = torch.cat(class_list)
            
            ds_name = os.path.join(SAVE_PATH, f"{model}_ssim_{ssim_coef}_class_{class_coef}_tv_{tv_coef}_merged.pt")
            print("dataset name:", ds_name)
            print("images:",images.shape)
            print("classes:",class_labels.shape)
            
            dataset = TensorDataset(images, class_labels)
            torch.save(dataset,ds_name)
            