import argparse
import logging
import os
import sys
import torch
import torch.nn as nn

import utils as utils
from quant_utils.quant_modules import *

import timm
from timm.data import resolve_data_config

from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention
from einops import rearrange

import random
import numpy as np

from image_gen_aug_utils import ColorJitter, GaussianNoise

def set_logger(save_dir, prefix=0):
    logger = logging.getLogger('baseline')
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    # file log
    file_handler = logging.FileHandler(os.path.join(save_dir, f"image_gen_{prefix}.log"))
    file_handler.setFormatter(file_formatter)
    
    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.setLevel(logging.INFO)
    return logger

class hook_class():
    def __init__(self):
        self.matmul_output = []
        
    def hook_fn_forward(self,module, input, output):
        self.matmul_output.append(output)

def get_image_prior_losses(inputs_jit):
    # Compute total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3)/np.sqrt(2) + torch.norm(diff4)/np.sqrt(2)
    return loss_var_l2

def ssim_loss_att_map_swin(att_hook_output, mode='full_map', clamp=True, self_zero=False): # cls, sptl, all, sptl_rand
    
    ssim_map = []
    
    for att in att_hook_output:
        att_map = rearrange(att, "B H N d -> (B N) H d")
        att_map = att_map - att_map.min(-1, keepdim=True)[0]
        att_map = att_map / (att_map.max(-1, keepdim=True)[0] + 1e-8)
        
        att_map = rearrange(att_map, "B H (w h) -> B H w h", w=7, h=7)
        mu = att_map.mean([-2,-1]) #(B H)
        
        mu_sq = (mu.pow(2)) # (B H)
        mu1_mu2 = att_map.mean([-2,-1]).unsqueeze(1)*att_map.mean([-2,-1]).unsqueeze(2) #(B H H)

        sigma_sq = (att_map*att_map).mean([-2,-1])-mu_sq #.sum(dim=[-2,-1]) #(B H)
        sigma12 = (att_map.unsqueeze(1)*att_map.unsqueeze(2)).mean([-2,-1])-mu1_mu2 #(B H H) 

        C1 = 0.01**2
        C2 = 0.03**2
        
        one_layer_ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu_sq.unsqueeze(1) + mu_sq.unsqueeze(2) + C1)*(sigma_sq.unsqueeze(1) + sigma_sq.unsqueeze(2) + C2))
        
        if mode == "full_map":
            ssim_map.append(one_layer_ssim_map)
        elif mode == 'loss':
            ssim_map.append((1-one_layer_ssim_map**2).mean())
        else:
            assert('invalid mode for SSIM')
    return ssim_map
        
        

def ssim_loss_att_map(att_hook_output, mode='full_map', clamp=True, self_zero=False): # cls, sptl, all, sptl_rand
    hook_stack = torch.stack(att_hook_output)[:,:,:,1:,1:197] # L B H Q-1 (w h)
    hook_stack = rearrange(hook_stack, "L B H N d -> L (B N) H d")
    hook_stack = hook_stack - hook_stack.min(-1, keepdim=True)[0]
    hook_stack = hook_stack / (hook_stack.max(-1, keepdim=True)[0] + 1e-8)

    
    att_map = rearrange(hook_stack, "L B H (w h) -> L B H w h", w=14,h=14)
    mu = att_map.mean([-2,-1]) #(L B H)

    mu_sq = (mu.pow(2)) # (L B H)
    mu1_mu2 = att_map.mean([-2,-1]).unsqueeze(2)*att_map.mean([-2,-1]).unsqueeze(3) #(L B H H)
    
    sigma_sq = (att_map*att_map).mean([-2,-1])-mu_sq  #(L B H)
    sigma12 = (att_map.unsqueeze(2)*att_map.unsqueeze(3)).mean([-2,-1])-mu1_mu2 #(L B H H) 

    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu_sq.unsqueeze(2) + mu_sq.unsqueeze(3) + C1)*(sigma_sq.unsqueeze(2) + sigma_sq.unsqueeze(3) + C2))
    if mode == 'loss':
        return ssim_map.mean() 
    elif mode == 'full_map':
        if self_zero:
            ssim_map = ssim_map - torch.eye(ssim_map.shape[-1], device=ssim_map.device).unsqueeze(0).unsqueeze(0)
        return ssim_map # L B H H
    else:
        assert('invalid mode for SSIM')
        
        

def generate(args, save_dir, logger):
    
    
    model_origin = timm.create_model(args.model, pretrained=True).eval()
    
    model_stats = resolve_data_config(dict(),model=model_origin, verbose=True)
    
    mu = torch.tensor(model_stats["mean"],device='cuda')
    std = torch.tensor(model_stats["std"],device='cuda')
    
    model_origin = model_origin.cuda()
    prepare_vit_model(model_origin)
    
    IMG_SIZE = model_origin.patch_embed.img_size[0]
    PATCH_SIZE = model_origin.patch_embed.patch_size[0]
    NUM_PATCHES = model_origin.patch_embed.num_patches
    
    aug_list = []
    if not args.no_gs:
        aug_list.append(GaussianNoise(args.batch_size, True, 0.5, args.iter))
    if not args.no_cj:
        aug_list.append(ColorJitter(args.batch_size, True))
        
    if not aug_list:
        aug_list.append(nn.Identity())
        
    aug = torch.nn.Sequential(*aug_list)
    
    
    if 'swin' in args.model:
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
    
    hook = hook_class()
    hook_hydra = hook_class()

    for m in model_origin.modules():
        if isinstance(m, Attention):
            m.matmul1.register_forward_hook(hook.hook_fn_forward)
        if isinstance(m, WindowAttention):
            m.matmul1.register_forward_hook(hook.hook_fn_forward)
            
            
    for batch in range(args.num_images//args.batch_size):
        img_train = torch.randn((args.batch_size,3,IMG_SIZE,IMG_SIZE)).cuda()
        img_train.requires_grad = True
        opt = torch.optim.AdamW([img_train], lr=0.1, betas=[0.9,0.999], eps=1e-8, weight_decay=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iter, 0.001)

        class_label = torch.randint(0,1000,(args.batch_size,)).cuda()
        
        if args.some_class == 1:
            class_label = torch.randint(0,10,(args.batch_size,)).cuda()
        elif args.some_class == 100:
            class_label = torch.randint(0,10,(args.batch_size,)).cuda() * 100
        
        NAN_FLAG = False
        
        for i in range(args.iter):
            
            if i < args.iter/2:
                JITTER = args.jitter
            else:
                JITTER = args.jitter*2
            # JITTER = 64
            hook.matmul_output.clear() 
            hook_hydra.matmul_output.clear()
            opt.zero_grad()
            
            lim_0, lim_1 = JITTER, JITTER
            
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            img_jit = torch.roll(img_train, shifts=(off1, off2), dims=(2, 3))
            
            if args.flip and random.random() > 0.5:
                img_jit = torch.flip(img_jit, dims=(3,))
                
            img_jit = aug(img_jit)
            
            output = model_origin(img_jit)
            class_loss = torch.zeros(1).cuda()
            ssim_loss = torch.zeros(1).cuda()
            
            if "swin" in args.model:
                ssim_loss = torch.stack(ssim_loss_att_map_swin(hook.matmul_output, mode='loss')).mean()
            else:
                ssim_map = ssim_loss_att_map(hook.matmul_output, mode="full_map")
                
                ssim_loss = (1-(ssim_map)**2).mean()

            
            class_loss = F.cross_entropy(output,class_label)
            _, pred_output = torch.max(output,dim=-1)
            correct_class = (pred_output == class_label).sum()
            
            loss_tv = get_image_prior_losses(img_train)
            
            total_loss = args.ssim_coef*ssim_loss + args.class_coef*class_loss + args.tv_coef*loss_tv
            
            if (i+1)%100 == 0:
                logger.info(f"[BATCH: {batch}/{args.num_images//args.batch_size}] [ITER: {i+1}/{args.iter}] [SSIM_LOSS: {ssim_loss.item():.3f}] [CL_LOSS: {class_loss.item():.3f}] [TV_LOSS: {loss_tv.item():.3f}] [CL_ACC: {correct_class.item()*100/args.batch_size:.2f}]")
            total_loss.backward()
            opt.step()
            lr_scheduler.step()
            
            
            if args.clamp_training:
                img_train.data = torch.clamp(img_train.data, (-mu/std).view(-1,1,1), ((1-mu)/std).view(-1,1,1))
            
            if torch.isnan(img_train).sum()>0:
                logger.info("NAN OCCURS, SKIP: ",torch.isnan(img_train).sum())
                NAN_FLAG=True
                break
                
            
        # image
        if not NAN_FLAG:
            torch.save(img_train.detach().cpu(), os.path.join(save_dir,f"{args.save_prefix+batch}_images.pt"))
            # class labels
            torch.save(class_label.detach().cpu(), os.path.join(save_dir,f"{args.save_prefix+batch}_class_labels.pt"))


def main():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--model', type=str, metavar='model',
                        help='target vit model', default="vit_base_patch16_224")
    parser.add_argument('--num_images', type=int, default=10000)
    parser.add_argument('--save_prefix', type=int, default=0)
    parser.add_argument('--iter', type=int, default=2000)

    parser.add_argument('--ssim_coef', type=float, default=1.0)
    parser.add_argument('--jitter', type=int, default=8)
    parser.add_argument('--flip', action="store_true")
    parser.add_argument('--class_coef', type=float, default=1.0)
    parser.add_argument('--tv_coef', type=float, default=2.5e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='./gen_images_raw')
    parser.add_argument('--clamp_training', action="store_true")
    parser.add_argument('--some_class', type=int, default=None)
    parser.add_argument('--mode', type=str, default="ssim")
    
    parser.add_argument('--no_gs', action="store_true")
    parser.add_argument('--no_cj', action="store_true")
    

    args = parser.parse_args()
    # print(args)

    save_dir = os.path.join(args.save_path,f"{args.model}_ssim_{args.ssim_coef}_class_{args.class_coef}_tv_{args.tv_coef}")
    if args.some_class == 1:
        save_dir = os.path.join(args.save_path,f"{args.model}_ssim_{args.ssim_coef}_class_{args.class_coef}_tv_{args.tv_coef}_0_9")
    elif args.some_class == 100:
        save_dir = os.path.join(args.save_path,f"{args.model}_ssim_{args.ssim_coef}_class_{args.class_coef}_tv_{args.tv_coef}_000_900")

    os.makedirs(save_dir, exist_ok=True)
    
    logger = set_logger(save_dir, args.save_prefix)
    logger.info(args)
    
    generate(args, save_dir,logger)


if __name__ == '__main__':
    main()