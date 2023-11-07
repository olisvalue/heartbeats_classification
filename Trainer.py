from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.optim import AdamW

#distributed training
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP

#custom
from utils.StatsMaker import StatisticsMaker
from Evaluater import Evaluater

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str, 
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        is_distributed: bool,
        main_gpu_id = 0,
        accum_iter = 1,
        without_eval=False
    ) -> None:
        self.gpu_id = gpu_id
        self.without_eval = without_eval
        self.is_distributed = is_distributed
        if is_distributed:
            self.stats = StatisticsMaker(model_name, gpu_id, main_gpu_id)
        else:
            self.stats = StatisticsMaker(model_name, gpu_id, gpu_id)
        # print("trying initialization of dataloaders")
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.accum_iter = accum_iter
        self.count_iter = 0

        self.optimizer = optimizer

        self.eval_every = 1
        self.save_every = 1

        self.model = model.to(gpu_id)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[gpu_id], find_unused_parameters=True)#,static_graph=True)
            self.module = self.model.module
        else:
            self.module = self.model
        
        self.evaluater = Evaluater(self.model, self.test_dataloader, self.gpu_id)


    def _run_batch(self, audio, targets):
        logits = self.model(audio)

        loss = nnf.cross_entropy(logits, targets)
        loss = loss / self.accum_iter
        self.count_iter += 1

        self.stats.add_loss(loss.item()*self.accum_iter)
        loss.backward()

        if self.count_iter == self.accum_iter:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.count_iter = 0


    def _run_train_epoch(self, epoch):
        if self.is_distributed:
            self.train_dataloader.sampler.set_epoch(epoch)
        self.model.train() 
        for audio, targets in self.stats.make_pbar(self.train_dataloader):
            audio = audio.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(audio, targets)
            self.stats.set_description()
        self.stats.save_epoch_loss()

    def _run_eval(self, epoch):
        # if self.is_distributed:
        #     self.test_dataloader.sampler.set_epoch(epoch) 
        metrics = self.evaluater.eval()         
        self.stats.process_metrics(self.module, metrics)    

    def train(self, max_epochs: int, start_epoch = 0):
        for epoch in range(start_epoch, max_epochs):
            self.stats.set_epoch(epoch)
            self.stats.epoch_time_measure()
            self._run_train_epoch(epoch)
            if epoch % self.save_every == 0:
                self.stats.save_last_params(self.module)
            if epoch % self.eval_every == 0 and not self.without_eval:
                self._run_eval(epoch)

            self.stats.epoch_time_measure()
            
    def evaluate(self):
        self.evaluater.eval()
    