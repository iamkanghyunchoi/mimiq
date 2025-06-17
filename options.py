import os
import shutil

from pyhocon import ConfigFactory

from utils.opt_static import NetOption


class Option(NetOption):    
    def __init__(self, conf_path, args):
        super(Option, self).__init__()
        self.conf = ConfigFactory.parse_file(conf_path)
        #  ------------ General options ----------------------------------------
        self.save_path = self.conf['save_path']
        self.dataPath = self.conf['dataPath']  # path for loading data set
        self.dataset = self.conf['dataset']  # options: imagenet | cifar100
        self.nGPU = self.conf['nGPU']  # number of GPUs to use by default
        self.GPU = self.conf['GPU']  # default gpu to use, options: range(nGPU)
        self.visible_devices = args.gpu #self.conf['visible_devices']
        self.network = self.conf['network']
        
        # ------------- Data options -------------------------------------------
        self.nThreads = self.conf['nThreads']  # number of data loader threads
        
        # ---------- Optimization options --------------------------------------
        # self.nEpochs = self.conf['nEpochs']  # number of total epochs to train
        if args.bs == None:
            self.batchSize = self.conf['batchSize']  # mini-batch size
        else:
            self.batchSize = args.bs  # mini-batch size
        self.momentum = self.conf['momentum']  # momentum
        if args.wd == None:
            self.weightDecay = float(self.conf['weightDecay'])  # weight decay
        else:
            self.weightDecay = float(args.wd)  # weight decay
        self.opt_type = self.conf['opt_type']
        self.warmup_epochs = self.conf['warmup_epochs']  # number of epochs for warmup

        if args.lrs == None:
            self.lr_S = self.conf['lr_S']  # initial learning rate
        else:
            self.lr_S = args.lrs  # initial learning rate
            
        if args.lr_policy == None:
            self.lrPolicy_S = self.conf['lrPolicy_S']
        else:
            self.lrPolicy_S = args.lr_policy
            
        if args.lr_step == None:
            self.step_S = self.conf['step_S']  # step for linear or exp learning rate policy
        else:
            self.step_S = args.lr_step
            
        
        self.decayRate_S = self.conf['decayRate_S']  # lr decay rate
        
        
        # ---------- Quantization options ---------------------------------------------
        if args.qw == None:
            self.qw = self.conf['qw']
        else:
            self.qw = args.qw

        if args.qa == None:
            self.qa = self.conf['qa']
        else:
            self.qa = args.qa
            
        # ---------- Model options ---------------------------------------------
        self.experimentID = f"DFViT_img_opt_{self.network}_qwqa_{self.qw}_{self.qa}_lrS_{self.lr_S}_T_{args.kl_temp}_{self.step_S}_wa_{args.wq_mode}_{args.aq_mode}_hd_{args.head_dist_distance}_{args.head_dist_coef}_img_ssim_{args.img_opt_ssim}_cls_{args.img_opt_cls}_tv_{args.img_opt_tv}_ns_{args.num_samples}_rand_{args.random_samples}_real_{args.real_data}_bs_{args.bs}_id_{self.conf['experimentID']}"
        self.nClasses = self.conf['nClasses']  # number of classes in the dataset
            
        # ----------KD options ---------------------------------------------
        self.temperature = args.kl_temp #self.conf['temperature']
        self.alpha = self.conf['alpha']
        self.ce_scale = args.ce_scale
        self.kd_scale = args.kd_scale
        
        # ----------Generator options ---------------------------------------------
        self.latent_dim = self.conf['latent_dim']
        self.img_size = self.conf['img_size']
        self.channels = self.conf['channels']

        self.lr_G = self.conf['lr_G']
        self.lrPolicy_G = self.conf['lrPolicy_G']  # options: multi_step | linear | exp | const | step
        self.step_G = self.conf['step_G']  # step for linear or exp learning rate policy
        self.decayRate_G = self.conf['decayRate_G']  # lr decay rate

        self.b1 = self.conf['b1']
        self.b2 = self.conf['b2']

        self.img_opt_cls = args.img_opt_cls
        self.img_opt_ssim = args.img_opt_ssim
        self.img_opt_tv = args.img_opt_tv
        self.dataset_path = args.dataset_path
        self.num_samples = args.num_samples
        self.random_samples = args.random_samples

        # ----------ETC ---------------------------------------------
        self.local = args.local
        self.head_dist_coef = args.head_dist_coef
        self.head_dist_distance = args.head_dist_distance
        self.save_model = args.save_model
        self.real_data = args.real_data

        self.wq_mode = args.wq_mode
        self.aq_mode = args.aq_mode

        self.kl_temp = args.kl_temp
        
        self.cache_dataset = args.cache_dataset
        
        self.lsq_g_scale = args.lsq_g_scale
        
        self.grad_acc = args.grad_acc
        self.batchSize = self.batchSize // self.grad_acc


        
    def set_save_path(self):
        self.save_path = self.save_path + "log_{}/".format(
            self.experimentID)
        
        if os.path.exists(self.save_path):
            print("{} file exist!".format(self.save_path))
            # action = input("Select Action: d (delete) / q (quit):").lower().strip()
            # act = action
            # if act == 'd':
            # 	shutil.rmtree(self.save_path)
            # else:
            # 	raise OSError("Directory {} exits!".format(self.save_path))
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def paramscheck(self, logger):
        logger.info("|===>The used PyTorch version is {}".format(
                self.torch_version))
        
        if self.dataset in ["cifar10", "mnist"]:
            self.nClasses = 10
        elif self.dataset == "cifar100":
            self.nClasses = 100
        elif self.dataset == "imagenet" or "thi_imgnet":
            self.nClasses = 1000
        elif self.dataset == "imagenet100":
            self.nClasses = 100