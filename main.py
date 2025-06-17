import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# option file should be modified according to your expriment
from options import Option

from dataloader import DataLoader
from trainer import Trainer

import utils as utils
from utils.arglist import *
from quant_utils.quant_modules import *

import timm
from timm.data import resolve_data_config

from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention


class ExperimentDesign:
    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.model_teacher = None
        
        self.optimizer_state = None
        self.trainer = None
        self.start_epoch = 0
        self.test_input = None

        self.unfreeze_Flag = True
        
        if self.settings.local:
            os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.visible_devices
        
        self.settings.set_save_path()
        self.logger = self.set_logger()
        self.settings.paramscheck(self.logger)

        self.prepare()
    
    def set_logger(self):
        logger = logging.getLogger('baseline')
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        # file log
        file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
        file_handler.setFormatter(file_formatter)
        
        # console log
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.setLevel(logging.INFO)
        return logger

    def prepare(self):
        self._set_gpu()
        self._set_model()
        self._set_dataloader()
        self._replace()
        self.logger.info(self.model)
        self._set_trainer()
    
    def _set_gpu(self):
        torch.manual_seed(self.settings.manualSeed)
        torch.cuda.manual_seed(self.settings.manualSeed)
        assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
        cudnn.benchmark = True

    def _set_dataloader(self):
        data_config = resolve_data_config(
            vars(self.settings),
            model=self.model,
            verbose=True
        )
        self.settings.data_config = data_config

        data_loader = DataLoader(dataset=self.settings.dataset,
                            batch_size=self.settings.batchSize,
                            data_path=self.settings.dataPath,
                            n_threads=self.settings.nThreads,
                            ten_crop=self.settings.tenCrop,
                            logger=self.logger,
                            cached=self.settings.cache_dataset,
                            stats=data_config,
                            grad_acc=self.settings.grad_acc)
        
        
        if self.settings.real_data:
            self.train_loader, self.test_loader = data_loader.getloader()

            self.settings.nEpochs = 100
            print("train dataset", len(self.train_loader.dataset))
        else:
            _, self.test_loader = data_loader.getloader()

            
            dataset_path = os.path.join(self.settings.dataset_path, f"{self.settings.network}_ssim_{self.settings.img_opt_ssim}_class_{self.settings.img_opt_cls}_tv_{self.settings.img_opt_tv}_merged.pt")
            dataset_full = torch.load(dataset_path)
                
            if len(dataset_full) > self.settings.num_samples:
                dataset, _ = torch.utils.data.random_split(dataset_full,[self.settings.num_samples, len(dataset_full)-self.settings.num_samples])
            else:
                dataset = dataset_full
                
            self.train_loader = torch.utils.data.DataLoader(dataset, batch_size = self.settings.batchSize,
                                                            num_workers=self.settings.nThreads, shuffle=True)
            
        print("validation dataset",len(self.test_loader.dataset))
        if self.settings.cache_dataset:
            for _ in self.test_loader:
                pass
            self.test_loader.dataset.set_use_cache(True)
        
        self.settings.patch_size = self.model.patch_embed.patch_size[0]
        self.settings.embed_dim = (self.settings.patch_size ** 2) * self.settings.channels
        
        if "swin" in self.settings.network:
            self.settings.model_num_head = []
            self.settings.model_num_windows = []

            for m in self.model.modules():
                if isinstance(m, WindowAttention):
                    self.settings.model_num_head.append(m.num_heads)
                    self.settings.model_num_windows.append((self.model.num_features // m.dim) ** 2)
                    self.settings.window_area = m.window_area
            self.settings.model_depth = len(self.settings.model_num_head)
        else:
            self.settings.model_depth = len(self.model.blocks)
            self.settings.model_num_head = self.model.blocks[0].attn.num_heads
            self.settings.seq_len = self.settings.img_size // self.settings.patch_size


    def _set_model(self):
        self.test_input = Variable(torch.randn(1, 3, 224, 224).cuda())
        self.model = timm.create_model(self.settings.network, pretrained=True)
        self.model.eval()
        self.model_teacher = timm.create_model(self.settings.network, pretrained=True)
        self.model_teacher.eval()


    def _set_trainer(self):
        # set lr master
        lr_master_S = utils.LRPolicy(self.settings.lr_S,
                                self.settings.nEpochs,
                                self.settings.lrPolicy_S)

        params_dict_S = {
            'step': self.settings.step_S,
            'decay_rate': self.settings.decayRate_S
        }
        
        lr_master_S.set_params(params_dict=params_dict_S)

        # set trainer
        self.trainer = Trainer(
            model=self.model,
            model_teacher=self.model_teacher,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            lr_master_S=lr_master_S,
            settings=self.settings,
            logger=self.logger,
            opt_type=self.settings.opt_type,
            optimizer_state=self.optimizer_state,
            run_count=self.start_epoch)
    
    def _replace(self):
        prepare_vit_model(self.model)
        if self.settings.head_dist_coef > 0.0:
            prepare_vit_model(self.model_teacher)
        quantize_model(self.model, qw=self.settings.qw, qa=self.settings.qa, 
                       weight_q_mode=self.settings.wq_mode, act_q_mode=self.settings.aq_mode,
                       lsq_g_scale=self.settings.lsq_g_scale)
        set_first_last_layer(self.model)


    def run(self):
        best_top1 = 100
        best_top5 = 100
        start_time = time.time()

        test_error, test_loss, test5_error = self.trainer.test_teacher(0)

        try:
            for epoch in range(self.start_epoch, self.settings.nEpochs):
                self.epoch = epoch
                self.start_epoch = 0
                
                if self.settings.real_data or self.settings.aq_mode == "lsq":
                    LESS_UNFREEZE=3
                else:
                    LESS_UNFREEZE=0

                if epoch < self.settings.warmup_epochs-1-LESS_UNFREEZE:
                    print ("\n self.unfreeze_model(self.model)\n")
                    unfreeze_model(self.model)
                to_train_mode(self.model) ####
                if self.settings.random_samples:
                    train_error, train_loss, train5_error = self.trainer.train_random(epoch=epoch)
                else:
                    train_error, train_loss, train5_error = self.trainer.train(epoch=epoch)

                freeze_model(self.model)
                to_eval_mode(self.model) ####

                if epoch > self.settings.warmup_epochs - 2 -LESS_UNFREEZE:
                    test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
                else:
                    test_error = 100
                    test5_error = 100


                if best_top1 >= test_error:
                    best_top1 = test_error
                    best_top5 = test5_error
                    best_ep = epoch
                    if self.settings.save_model:
                        torch.save(self.model.state_dict(),os.path.join(self.settings.save_path,"best_model.pt"))
                    
                
                self.logger.info("#==>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}".format(best_top1, best_top5))
                self.logger.info("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f} at ep: {:d}".format(100 - best_top1,
                                                                                                    100 - best_top5, best_ep))

            
            message = self.settings.experimentID+"\n"
            message += "#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f} at ep {:d}\n".format( 100 - best_top1,
                                                                                                    100 - best_top5, best_ep)

        except BaseException as e:
            self.logger.error("Training is terminating due to exception: {}".format(str(e)))
            message = self.settings.experimentID+"\n"
            message += "Training is terminating due to exception: {}".format(str(e))
            traceback.print_exc()
        
        end_time = time.time()
        time_interval = end_time - start_time
        t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
        self.logger.info(t_string)

        return best_top1, best_top5


def main():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--conf_path', type=str, metavar='conf_path',
                        help='input the path of config file')
    parser.add_argument('--id', type=int, metavar='experiment_id',
                        help='Experiment ID')
    parser.add_argument('--ce_scale', type=float, default=0.0)
    parser.add_argument('--kd_scale', type=float, default=1.0)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--lrs',type=float, default=None)
    parser.add_argument('--lr_policy', type=str, default=None)
    parser.add_argument('--lr_step', type=arg_as_list, default=None)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--local', action="store_true")
    parser.add_argument('--qw', type=int, default=None)
    parser.add_argument('--qa', type=int, default=None)
    parser.add_argument('--head_dist_coef', type=float, default=10.0)
    parser.add_argument('--head_dist_distance', type=str, default="ssim") # mse | kl | ssim (ours)
    parser.add_argument('--img_opt_cls', type=float, default=1.0)
    parser.add_argument('--img_opt_ssim', type=float, default=1.0)
    parser.add_argument('--img_opt_tv', type=float, default=2.5e-5)

    parser.add_argument('--dataset_path', type=str, default="/datasets/vit_img_opt_ssim_dataset")
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--random_samples', action="store_true")

    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--real_data', action="store_true")
    parser.add_argument('--wq_mode', type=str, default='minmax')
    parser.add_argument('--aq_mode', type=str, default='lsq')
    parser.add_argument('--bs', type=int, default=None)
    
    parser.add_argument('--kl_temp', type=float, default=5.0)
    
    parser.add_argument('--cache_dataset', action="store_true")
    
    parser.add_argument('--lsq_g_scale', type=float, default=0.01)
    parser.add_argument('--grad_acc', type=int, default=1)
    
    args = parser.parse_args()

    print(args)
    
    option = Option(args.conf_path, args)
    option.manualSeed = args.id + 1
    option.experimentID = option.experimentID + "{:0>2d}_repeat".format(args.id)

    experiment = ExperimentDesign(option)
    experiment.run()


if __name__ == '__main__':
    main()
