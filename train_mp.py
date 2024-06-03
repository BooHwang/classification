#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File      : train_mp.py
@Time      : 2024/06/03 17:20:22
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''

import os
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler as GradScaler

from nets import get_model_from_name
from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import (download_weights, get_classes, get_lr_scheduler,
                         set_optimizer_lr, show_config, weights_init)
from utils.utils_fit import fit_one_epoch
import warnings
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, rank, world_size):
        classes_path               = 'model_data/cls_classes.txt' 
        self.class_names, self.num_classes = get_classes(classes_path)
        
        self.backbone              = "mobilenetv2"
        self.input_shape           = [256, 256]
        self.pretrained            = False
        self.train_annotation_path = "face_train.txt"
        self.test_annotation_path  = 'face_test.txt'
        self.fp16                  = False 
        self.lr_decay_type         = "cos"
        self.save_period           = 1
        self.save_dir              = 'logs/face_0602'
        self.num_workers           = 0
        self.epoch_start           = 0
        self.epoch_end             = 200
        self.batch_size            = 128  
        
        self.Init_lr               = 1e-3
        self.Min_lr                = self.Init_lr * 0.01
          
        self.optimizer_type        = "adam"
        self.momentum              = 0.9
        self.weight_decay          = 0

        self.world_size            = world_size
        
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        # self.init_loss()
        # self.init_writer()
        self.train()
        self.cleanup()
        
    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        os.environ['MASTER_ADDR'] = "localhost"
        os.environ['MASTER_PORT'] = "12255"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def init_model(self):
        if self.pretrained:
            download_weights(self.backbone)
        
        self.model = get_model_from_name[self.backbone](num_classes = self.num_classes, pretrained = self.pretrained)
        
        if self.rank == 0:
            self.loss_history = LossHistory(self.save_dir, self.model, input_shape=self.input_shape)
        else:
            self.loss_history = None
            
        self.model = self.model.to(self.rank)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model_ddp = DDP(self.model, device_ids=[self.rank], find_unused_parameters=False)
        
        if self.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            
        nbs             = 64
        lr_limit_max    = 1e-3 if self.optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if self.optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(self.batch_size / nbs * self.Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(self.batch_size / nbs * self.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        self.optimizer = {
            'adam'  : optim.Adam(self.model.parameters(), Init_lr_fit, betas = (self.momentum, 0.999), weight_decay=self.weight_decay),
            'sgd'   : optim.SGD(self.model.parameters(), Init_lr_fit, momentum = self.momentum, nesterov=True)
        }[self.optimizer_type]
        
        self.lr_scheduler_func = get_lr_scheduler(self.lr_decay_type, Init_lr_fit, Min_lr_fit, self.epoch_end)
        
    
    def init_datasets(self):
        with open(self.train_annotation_path, encoding='utf-8') as f:
            train_lines = f.readlines()
        with open(self.test_annotation_path, encoding='utf-8') as f:
            val_lines   = f.readlines()
        self.num_train   = len(train_lines)
        self.num_val     = len(val_lines)
        np.random.seed(10101)
        np.random.shuffle(train_lines)
        np.random.seed(None)
        
        train_dataset   = DataGenerator(train_lines, self.input_shape, self.rank, True)
        val_dataset     = DataGenerator(val_lines, self.input_shape, self.rank, False)
        
        self.train_sampler   = DistributedSampler(train_dataset, shuffle=True,)
        self.val_sampler     = DistributedSampler(val_dataset, shuffle=False,)
        batch_size      = self.batch_size // self.world_size
        
        self.gen             = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True, 
                                drop_last=True, collate_fn=detection_collate, sampler=self.train_sampler)
        self.gen_val         = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate, sampler=self.val_sampler)

    def train(self):
        for epoch in range(self.epoch_start, self.epoch_end):
            self.train_sampler.set_epoch(epoch)
            
            epoch_step      = self.num_train // self.batch_size
            epoch_step_val  = self.num_val // self.batch_size
                
            set_optimizer_lr(self.optimizer, self.lr_scheduler_func, epoch)
            
            fit_one_epoch(self.model_ddp, self.model, self.loss_history, self.optimizer, epoch, epoch_step, epoch_step_val, self.gen, self.gen_val, self.epoch_end, True, self.fp16, self.scaler, self.save_period, self.save_dir, self.rank)

        if self.rank == 0:
            self.loss_history.writer.close()
            
    def cleanup(self):
        dist.destroy_process_group()
            
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"  
    # os.environ["NCCL_DEBUG"] = "INFO"

    world_size = torch.cuda.device_count()
    spawn(Trainer, args=(world_size, ), nprocs=world_size, join=True)
