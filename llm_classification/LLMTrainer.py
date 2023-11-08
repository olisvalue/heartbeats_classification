from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from tqdm import tqdm
import time
import datetime
import os
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.optim import AdamW

# For evaluation
from typing import MutableMapping, MutableSequence,\
    Any, Union, List, Dict, Tuple
from llm_classification.eval_metrics import evaluate_metrics
from terminaltables import AsciiTable
import pickle

#distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset


class Timer:
    def __init__(self, save_path):
        self.start_time = None
        self.measured_time = {}
        self.save_path = save_path
    def time_measure(self, epoch):
        if self.start_time is None:
            self.start_time = time.time()
        else:
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            self.measured_time[epoch] = elapsed_time
            self.start_time = None
            self.save_info()
    def save_info(self):
         with open(self.save_path, 'w') as file:
             file.write(f'total time is {sum([time for time in self.measured_time.values()])/60} mins\n')
             for epoch, time_value in self.measured_time.items():
                file.write(f'epoch {epoch}: {time_value/60} mins\n')

# Folder creation
class StatisticsMaker:
    def __init__(self, model_name, gpu_id, main_gpu_id):
        self.main_gpu_id = main_gpu_id
        self.given_gpu_id = gpu_id
        # self.pbar = None
        self.base_dir = "./data/Train_record/" + model_name + '/'
        self.params_dir = self.base_dir + "params/"
        self.csv_path = self.base_dir + "eval_scores.csv"
        self.best_model_path = self.base_dir + "best_model"
        self.epochs = []
        self.metrics_list = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor","rouge_l", "spice", "cider", "spider"]
        self.main_metric = "spider"
        self.scores_dict = defaultdict(dict)
        self.create_directory()
        self.epoch = -1
        self.epoch_losses = dict()
        self.loss = 0
        self.num_runs = 0
        self.timer = Timer(self.base_dir + "time_info.txt")
        #add results table

    def _active_method(func):
        def wrapper(self, *args, **kwargs):
            if self.main_gpu_id == self.given_gpu_id:
                return func(self, *args, **kwargs)
            else:
                # If GPU ID is not main, return None or handle it as needed
                return None
        return wrapper
         
    def _make_metrics_graph(self):
        plt.figure(figsize=(10, 6))
        for metric in self.metrics_list:
            plt.plot(*zip(*sorted(self.scores_dict[metric].items())), marker='o', label=metric, color=np.random.rand(3))
        plt.title('Evaluation while training')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics values')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.base_dir + "eval_scores.png")
    def _make_loss_graph(self):
        plt.figure(figsize=(10, 6))
        plt.plot(*zip(*sorted(self.epoch_losses.items())), marker='o', color=np.random.rand(3))
        plt.title('Losses while training')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        # plt.legend()
        plt.grid(True)
        plt.savefig(self.base_dir + "epoch_losses.png")

    def make_pbar(self, dataloader):
        if self.main_gpu_id == self.given_gpu_id:
            self.pbar = tqdm(dataloader, desc=f"Training Epoch {self.epoch}", position=0, leave=True)
            return self.pbar
        else:
            return dataloader

    @_active_method
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.epochs.append(epoch)
        self.num_runs = 0
        self.loss = 0

    @_active_method
    def add_loss(self, loss):
        self.loss += loss
        self.num_runs += 1

    @_active_method
    def set_description(self):
         self.pbar.set_description(f"Training Epoch {self.epoch}, Loss = {round(self.get_avg_loss(), 5)}")

    @_active_method
    def get_avg_loss(self):
        return self.loss/self.num_runs
    
    @_active_method
    def save_epoch_loss(self):
        if len(self.epoch_losses) == 0 and self.epoch != 0:
            df = pd.read_csv(self.base_dir + 'loss_info.csv')
            self.epoch_losses = df.to_dict(orient='dict')['0']
        self.epoch_losses[self.epoch] = self.loss/self.num_runs
        self._make_loss_graph()
        df = pd.DataFrame.from_dict(self.epoch_losses, orient='index')
        df.to_csv(self.base_dir + 'loss_info.csv', index=False)
    
    @_active_method
    def epoch_time_measure(self):
        self.timer.time_measure(self.epoch)

    @_active_method
    def create_directory(self):
        directory1 = self.params_dir
        try:
            if not os.path.exists(directory1):
                os.makedirs(directory1)
        except OSError:
            print("Error: Failed to create the directory.")

    @_active_method
    def save_epoch_params(self, model):
        param_file = self.params_dir + "epoch_" + str(self.epoch) + ".pt"
        torch.save(model.state_dict(), param_file)
    
    @_active_method
    def process_metrics(self, model, metrics_file):
        if len(self.scores_dict) == 0 and self.epoch != 0:
            df = pd.read_csv(self.csv_path)
            self.scores_dict = df.to_dict(orient='dict')
        for metric in self.metrics_list:
            score = metrics_file[metric]["score"]
            self.scores_dict[metric][self.epoch] = score
        #save scores to csv file
        df = pd.DataFrame.from_dict(self.scores_dict, orient='index').T
        df.to_csv(self.csv_path, index=False)

        self._make_metrics_graph()
        cur_score = self.scores_dict[self.main_metric][self.epoch]
        best_model = True if cur_score == max(self.scores_dict[self.main_metric].values()) else False
        if best_model:
            torch.save(model.state_dict(), self.best_model_path)
            short_log = f"Best model achieved on epoch {self.epoch}, with SPIDEr score = {cur_score}\n"
            with open(self.base_dir + "short_log.txt", 'w') as file:
                file.write(short_log)


class Trainer:
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
                                   self.is_distributed, self.test_dataloader.nb_references)

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
        for audio, tokens, mask, _ in self.stats.make_pbar(self.train_dataloader):
            audio = audio.to(self.gpu_id)
            tokens = tokens.to(self.gpu_id)
            mask = mask.to(self.gpu_id)
            self._run_batch(audio, tokens, mask)
            self.stats.set_description()
        self.stats.save_epoch_loss()

    def _run_eval(self, epoch):
        # if self.is_distributed:
        #     self.test_dataloader.sampler.set_epoch(epoch) 
        metrics, _, _ = self.evaluater.eval()         
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


    

class Evaluater:
    def __init__(self, model, dataloader, device, is_distributed = False, 
                 nb_reference_captions = 5, beam_search = True):
        self.beam_search = beam_search
        self.model = model
        self.dataloader = dataloader
        self.is_distributed = is_distributed
        self.device = device
        self.main_device = 0
        self.nb_reference_captions = nb_reference_captions 
        self.captions_pred: List[Dict] = []
        self.captions_gt: List[Dict] = []

    def _collect_distributed(self):
        world_size = dist.get_world_size()
        all_captions = [None] * world_size
        dist.all_gather_object(all_captions, [self.captions_pred, self.captions_gt]) 
        if self.device == self.main_device:
            captions_pred_lists = [captions[0] for captions in all_captions]
            captions_gt_lists = [captions[1] for captions in all_captions]
            captions_pred = []
            captions_gt = []
            for sublist in captions_pred_lists:
                captions_pred.extend(sublist)
            for sublist in captions_gt_lists:
                captions_gt.extend(sublist)
            self.captions_pred = captions_pred
            self.captions_gt = captions_gt
                   
    def _get_metrics(self):
        if self.is_distributed:
            self._collect_distributed()
            if self.device != self.main_device:
                return [None, None, None]
        metrics = evaluate_metrics(self.captions_pred, self.captions_gt, nb_reference_captions=self.nb_reference_captions)
        return [metrics, self.captions_pred, self.captions_gt]
    
    def _append_captions(self, pred_caption, captions, filenames):
        self.captions_pred.append({
                                'file_name': filenames[0], 
                                'caption_predicted': pred_caption})
        if self.nb_reference_captions == 5:
            self.captions_gt.append({
                                'file_name': filenames[0],
                                'caption_reference_01': captions[0],
                                'caption_reference_02': captions[1],
                                'caption_reference_03': captions[2],
                                'caption_reference_04': captions[3],
                                'caption_reference_05': captions[4]})
        elif self.nb_reference_captions == 1:
            self.captions_gt.append({
                                'file_name': filenames[0],
                                'caption_reference_01': captions[0]})
            
    def _eval_pack(self, audio, captions, filenames):
        with torch.no_grad():
            audio = audio[0].unsqueeze(0)
            audio = audio.to(self.device)
            if self.beam_search == True:
                pred_caption = self.model(audio, beam_search = True)[0][0]
            else:
                pred_caption = self.model(audio, beam_search = False)[0]
        self._append_captions(pred_caption, captions, filenames)

    def eval(self):
        self.model.eval().to(self.device)
        self.captions_pred = []
        self.captions_gt = []
        for audio, captions, filenames in tqdm(self.dataloader, desc="Eval using dataset..."):
            self._eval_pack(audio, captions, filenames)
        return self._get_metrics()    
            

def eval_model_parallel(model, test_dataloader, beam_search, device, test_captions = 5) :
    
    model.eval()
    model.to(device)

    captions_pred: List[Dict] = []
    captions_gt: List[Dict] = []
        
    for i, (audio, captions, f_names) in enumerate(tqdm(test_dataloader, desc="Eval using dataset...")):
        with torch.no_grad():
            audio = audio[0].unsqueeze(0)
            audio = audio.to(device)
            
            if beam_search == True :
                pred_caption = model(audio)[0][0]
            else :
                pred_caption = model(audio)[0]

        captions_pred.append({
                            'file_name': f_names[0], 
                            'caption_predicted': pred_caption})
        if test_captions == 5:
            captions_gt.append({
                                'file_name': f_names[0],
                                'caption_reference_01': captions[0],
                                'caption_reference_02': captions[1],
                                'caption_reference_03': captions[2],
                                'caption_reference_04': captions[3],
                                'caption_reference_05': captions[4]})
        elif test_captions == 1:
            captions_gt.append({
                                'file_name': f_names[0],
                                'caption_reference_01': captions[0]})
              
    
    world_size = dist.get_world_size()
    all_captions = [None] * world_size
    dist.all_gather_object(all_captions, [captions_pred, captions_gt]) 

    if device == 0:
        captions_pred_lists = [captions[0] for captions in all_captions]
        captions_gt_lists = [captions[1] for captions in all_captions]
        
        captions_pred = []
        captions_gt = []
        for sublist in captions_pred_lists:
            captions_pred.extend(sublist)
        for sublist in captions_gt_lists:
            captions_gt.extend(sublist)
        if test_captions == 1:
            metrics = evaluate_metrics(captions_pred, captions_gt, nb_reference_captions=1)
        elif test_captions == 5:
            metrics = evaluate_metrics(captions_pred, captions_gt, nb_reference_captions=5)
        return [metrics, captions_pred, captions_gt] 
    else:
        return [None, None, None]