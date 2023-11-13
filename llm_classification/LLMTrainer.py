import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.optim import AdamW

# For evaluation
from typing import MutableMapping, MutableSequence,\
    Any, Union, List, Dict, Tuple
from llm_classification.eval_metrics import evaluate_metrics

#distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset

from Evaluater import Evaluater
from utils.StatsMaker import StatisticsMaker

class LLMTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str, 
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        gpu_id: int,
        is_distributed: bool,
        main_gpu_id = 0,
        accum_iter = 1,
    ) -> None:
        self.gpu_id = gpu_id
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
        self.scheduler = scheduler
        self.epoch_freeze_encoder = 14
        self.eval_every = 1
        self.save_every = 1

        print("trying initialize the model")
        self.model = model.to(gpu_id)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[gpu_id])#, find_unused_parameters=True)#,static_graph=True)
            self.module = self.model.module
        else:
            self.module = self.model
        
        self.evaluater = Evaluater(self.model, self.test_dataloader, self.gpu_id,
                                   self.is_distributed, llm_mode=True)

        print("model initialized successfully")


    def _run_batch(self, audio, tokens, mask):
        logits = self.model(audio, tokens, mask)[:, self.module.prefix_length - 1: -1]
        
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        loss = loss / self.accum_iter
        self.count_iter += 1

        self.stats.add_loss(loss.item()*self.accum_iter)
        loss.backward()

        if self.count_iter == self.accum_iter:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.count_iter = 0


    def _run_train_epoch(self, epoch):
        # b_sz = len(next(iter(self.train_data))[0])
        # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        if self.is_distributed:
            self.train_dataloader.sampler.set_epoch(epoch)
        self.model.train()
        print("start training epoch loop")
        # pbar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}") 
        for audio, _, tokens, mask in self.stats.make_pbar(self.train_dataloader):
            audio = audio.to(self.gpu_id)
            tokens = tokens.to(self.gpu_id)
            mask = mask.to(self.gpu_id)
            self._run_batch(audio, tokens, mask)
            self.stats.set_description()
        self.stats.save_epoch_loss()

    def _run_eval(self, epoch):
        metrics = self.evaluater.eval()         
        self.stats.process_metrics(self.module, metrics)    

    def _freeze_encoder(self):
        for param in self.module.audio_encoder.parameters():
                    if param.requires_grad == True :
                        param.requires_grad = False
                        print("Encoder has been freezed")

    def train(self, max_epochs: int, start_epoch = 0):
        for epoch in range(start_epoch, max_epochs):
            self.stats.set_epoch(epoch)
            self.stats.epoch_time_measure()
            print("trying to run train epoch")
            self._run_train_epoch(epoch)
            # if epoch % self.save_every == 0:
            #     self.stats.save_epoch_params(self.module)
            if epoch % self.eval_every == 0:
                self._run_eval(epoch)
            # if epoch == self.epoch_freeze_encoder:
            #     self._freeze_encoder()
            self.stats.epoch_time_measure()

    def evaluate(self):
        self.evaluater.eval()
