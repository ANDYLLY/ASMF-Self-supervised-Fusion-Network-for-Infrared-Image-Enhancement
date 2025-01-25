import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import random
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir)
sys.path.append(current_dir)

from .network import network_ASM_main
import ASM_loader
import os
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time
import torchvision

class ASMS_Net(nn.Module):
    def __init__(self,
                 experiment="std",
                 # model params
                 nn_num_layer=8,
                 nn_num_channel=64,
                 nn_backbone="default",
                 nn_lowbound_enable=1,

                 #loss functions
                 W_T=100,
                 W_Dark=20,
                 W_G=1,
                 enable_T=1,
                 dataset="LGC",

                 # train setting
                 train_bn=4,
                 train_workers=16,
                 pretrained="",
                 lr=1e-2,
                 weight_decay=3e-5,
                 grad_clip=1,
                 # test setting
                 inference_iteration=6
                 ):
        super(ASMS_Net, self).__init__()

        # instantiate the model and loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.nn = network_ASM_main.ASM_S_Net(nn_num_layer,
                                    nn_num_channel,
                                    nn_backbone,
                                    nn_lowbound_enable,
                                    inference_iteration
                                    )

        self.loss = None
        # data preparation
        self.current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.current_file_dir)
        self.dataset = dataset
        self.train_dir = os.path.join(self.root_dir, 'data', 'train_data', self.dataset)
        self.test_dir = os.path.join(self.root_dir, 'data', 'test_data', self.dataset)
        self.check_dir = os.path.join(self.root_dir, 'data', 'check_data', self.dataset)
        print("load dataset train, check, test, valid, small_valid for ASM")

        self.test_bn=1
        self.train_data = ASM_loader.ASM_train_loader(IR_images_path=self.train_dir, minmax=True)
        self.check_data = ASM_loader.ASM_train_loader(IR_images_path=self.check_dir, minmax=True)
        self.test_data = ASM_loader.ASM_test_loader(IR_images_path=self.test_dir, minmax=True)
        self.valid_data = ASM_loader.ASM_test_loader(IR_images_path=self.train_dir, minmax=True)
        self.small_valid_data = ASM_loader.ASM_test_loader(IR_images_path=self.check_dir, minmax=True)

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=train_bn, shuffle=False,
                                                        num_workers=train_workers, pin_memory=False)
        self.check_loader = torch.utils.data.DataLoader(self.check_data, batch_size=1, shuffle=False,
                                                        num_workers=train_workers, pin_memory=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.test_bn, shuffle=False,
                                                       num_workers=train_workers, pin_memory=False)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_data, batch_size=self.test_bn, shuffle=False, num_workers=train_workers,
                                                        pin_memory=False)
        self.small_valid_loader = torch.utils.data.DataLoader(self.small_valid_data, batch_size=self.test_bn, shuffle=False,
                                                              num_workers=train_workers, pin_memory=False)

        if pretrained:
            self.pretrained = True
            self.pretrained_root = os.path.join(self.root_dir, 'experiment', pretrained, 'ASMS')
            self.pretrained_snapshots_dir = os.path.join(self.pretrained_root, 'snapshots')
            self.pretrained_snapshots_file = ASMS_Net.get_latest_weight_file(self.pretrained_snapshots_dir)
            self.pretrained_snapshots = os.path.join(self.pretrained_snapshots_dir, self.pretrained_snapshots_file)

        else:
            self.pretrained = False
            self.pretrained_root = None
            self.pretrained_snapshots_dir = None
            self.pretrained_snapshots_file = None
            self.pretrained_snapshots = None

        self.record_root_dir = os.path.join(self.root_dir, 'experiment', experiment, 'ASMS')
        self.record_snapshots = os.path.join(self.record_root_dir, 'snapshots')
        self.record_effect = os.path.join(self.record_root_dir, 'effect')
        self.record_run = os.path.join(self.record_root_dir, 'run')
        os.makedirs(self.record_root_dir, exist_ok=True)
        os.makedirs(self.record_snapshots, exist_ok=True)
        os.makedirs(self.record_effect, exist_ok=True)
        os.makedirs(self.record_run, exist_ok=True)

        self.record_result = os.path.join(self.record_root_dir, 'result')
        os.makedirs(self.record_result, exist_ok=True)

        self.record_valid_result = os.path.join(self.record_result, 'valid')
        self.record_check_result = os.path.join(self.record_result, 'check')
        self.record_test_result = os.path.join(self.record_result, 'test')

        os.makedirs(os.path.join(self.record_valid_result, "iter0"), exist_ok=True)
        os.makedirs(os.path.join(self.record_check_result, "iter0"), exist_ok=True)
        os.makedirs(os.path.join(self.record_test_result, "iter0"), exist_ok=True)

        self.inference_iteration = inference_iteration
        self.iteration_results_valid = ASMS_Net.create_iteration_folders(self.record_valid_result, self.inference_iteration)
        self.iteration_results_check = ASMS_Net.create_iteration_folders(self.record_check_result, self.inference_iteration)
        self.iteration_results_test = ASMS_Net.create_iteration_folders(self.record_test_result, self.inference_iteration)

    @staticmethod
    def get_latest_weight_file(weights_directory):
        files = os.listdir(weights_directory)
        pattern = re.compile(r'Epoch(\d+)\.pth')
        max_epoch = -1
        latest_file = None
        for file in files:
            match = pattern.match(file)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_file = file
        if latest_file is None:
            print(f"No weight files found in directory: {weights_directory}")
            exit(1)
        return latest_file

    @staticmethod
    def write_model_summary(file_path, params):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            for key, value in params.items():
                file.write(f"{key}={value}\n")

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)  
        torch.manual_seed(seed) 
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  
            torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False  

    @staticmethod
    def create_iteration_folders(base_path, num_folders):
        folder_names = []
        for i in range(1, num_folders + 1):
            folder_name = f"iter{i}"
            folder_path = os.path.join(base_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            folder_names.append(folder_path)
        return folder_names

    @staticmethod
    def initialize_weights_asms(model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1., 0.02)

    @staticmethod
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v 
            else:
                new_state_dict[k] = v
        return new_state_dict

    @staticmethod
    def load_pretrained_weights(model, pretrained_path):
        if pretrained_path:
            print('load weights')
            if os.path.isfile(pretrained_path):
                print(f"Loading pretrained weights from {pretrained_path}")
                model.load_state_dict(ASMS_Net.remove_module_prefix(torch.load(pretrained_path)))
            else:
                print(f"No pretrained weights found at {pretrained_path}")
        else:
            print("not found pth, wrong call")

    def get_inputdata(self):
        return self.test_loader

    def get_asm(self):
        ASMS_Net.load_pretrained_weights(self.nn, self.pretrained_snapshots)
        return self.nn

    def load_asm(self):
        ASMS_Net.load_pretrained_weights(self.nn, self.pretrained_snapshots)

    
        