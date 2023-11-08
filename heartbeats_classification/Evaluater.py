import torch
from tqdm import tqdm
import pandas as pd

class Evaluater:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.predicts = []
        self.ground_truth = []
        self.file_names = []

    def _get_accuracy(self):
        predicts = torch.cat(self.predicts, dim=0)
        ground_truth = torch.cat(self.ground_truth, dim=0)  
        correct, total = 0, len(predicts)

        print(predicts.shape, predicts)
        print(ground_truth.shape, ground_truth)
        for i in range(total):
            if predicts[i] == ground_truth[i]:
                correct += 1

        return correct/total
            
    def _eval_pack(self, audio, targets):
        with torch.no_grad():
            pred = self.model(audio)
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

        return self._get_accuracy()