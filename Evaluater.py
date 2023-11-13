import torch
from tqdm import tqdm
import pandas as pd
from torchmetrics import F1Score, Accuracy, Precision, Recall
import torch.distributed as dist

from data.adjectives import *
import string

class Evaluater:
    def __init__(self, model, dataloader, device, main_device=0, is_distributed=False, llm_mode=False):
        self.llm_mode=llm_mode

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.predicts = []
        self.ground_truth = []
        self.file_names = []

        self.main_device = main_device
        self.is_distributed=is_distributed

        self.f1 = F1Score(task="multiclass", num_classes=2, average="macro")
        self.acc = Accuracy(task="multiclass", num_classes=3)
        self.pr = Precision(task="multiclass", num_classes=2, average="macro")
        self.rc = Recall(task="multiclass", num_classes=2, average="macro")


        self.normal_words = NORMAL_ADJ
        self.abnormal_words = ABNORMAL_ADJ

    def _collect_distributed(self):
        world_size = dist.get_world_size()
        all_objects = [None] * world_size
        dist.all_gather_object(all_objects, [self.predicts, self.ground_truth]) 
        if self.device == self.main_device:
            predict_lists = [object[0] for object in all_objects]
            gt_lists = [object[1] for object in all_objects]
            predicts = []
            ground_truth = []
            for sublist in predict_lists:
                predicts.extend(sublist)
            for sublist in gt_lists:
                ground_truth.extend(sublist)
            self.predicts = predicts
            self.ground_truth = ground_truth
            
    def _get_metrics(self):
        predicts = torch.cat(self.predicts, dim=0)
        ground_truth = torch.cat(self.ground_truth, dim=0)

        if self.is_distributed:
            self._collect_distributed()
            if self.device != self.main_device:
                return [None, None, None]
    
        return {"f1": self.f1(predicts, ground_truth).item(),
                "accuracy": self.acc(predicts, ground_truth).item(),
                "precision": self.pr(predicts, ground_truth).item(),
                "recall": self.rc(predicts, ground_truth).item()}

    def clear_caption(self, caption):
        translator = str.maketrans("", "", string.punctuation)
        clean_text = caption.translate(translator)
        return clean_text.lower().split()
    
    def check_words_in_text(self, text: list[str], word_set: list[str]):
        for word in word_set:
            if word.lower() in text:
                return True
        return False

    def captions_to_labels(self, captions):
        pred = torch.ones(len(captions)) * 2 
        for i,caption in enumerate(captions):
            caption = self.clear_caption(caption)
            if self.check_words_in_text(caption, self.normal_words):
                pred[i] = 0
            if self.check_words_in_text(caption, self.abnormal_words):
                pred[i] = 1
        return pred       
    def _eval_pack(self, audio, targets):
        with torch.no_grad():
            pred = self.model(audio)
        
        if self.llm_mode==True:
            pred = self.captions_to_labels(pred)

        self.predicts.append(pred.detach().cpu())
        self.ground_truth.append(targets)
    

    def _save_preds(self, dir_path):
      
        df = pd.DataFrame({'filename': self.file_names, 'preds': self.predicts})
        tsv_file_path = dir_path + '/preds.tsv'

        df.to_csv(tsv_file_path, sep='\t', index=False)

    def get_predicts(self, dir_path):
        self.model.eval().to(self.device)
        for audio, file_names in tqdm(self.dataloader, desc="Calculating predictions..."):
            with torch.no_grad():
                pred = self.model(audio.to(self.device))
                self.predicts.extend(pred.detach().tolist())
                self.file_names.extend(file_names)
        self._save_preds(dir_path)
        return self.predicts, self.file_names

    def eval(self):
        self.model.eval().to(self.device)

        for audio, targets in tqdm(self.dataloader, desc="Evaluating..."):
            self._eval_pack(audio.to(self.device), targets)

        return self._get_metrics()