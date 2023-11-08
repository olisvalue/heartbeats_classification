import pandas as pd
import torchaudio, torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os

from collections import defaultdict

class MakeDataset(Dataset):
    def __init__(self, base_dir, split, SR=16000, set_length = 35, raw_audio=False, unlabeled_mode=False) :  # split = 'train' or 'test'
        super(MakeDataset, self).__init__()

        self.raw_audio=raw_audio
        self.unlabeled_mode=unlabeled_mode
        self.split = split
        self.set_length = set_length
        self.SR = SR

        self.data_dir = base_dir + '/' + split + '/'
        df = pd.read_csv(base_dir + '/dataset.csv')
        word_to_number = {"Normal": 0, "Abnormal": 1}
        df['label'] = df['label'].map(word_to_number)
        df_dict = df.set_index('id').to_dict()['label']

        self.path_list = []
        self.targets = []
        self.file_names = list(set(os.listdir(self.data_dir)))

        self.audio_files = []

        for file_name in tqdm(self.file_names, desc = f'get {split} dataset from {base_dir}...'):
            self.path_list.append(self.data_dir+file_name)
            if self.unlabeled_mode==False:
                # if df_dict[file_name[:-5]] == 1:
                #     continue
                self.targets.append(df_dict[file_name[:-5]])
            self.audio_files.append(self.process_audio_file(self.path_list[-1]))

    def __len__(self):
        return len(self.audio_files)
    
    def mel_transform(self, audio):
        transform = torchaudio.transforms.MelSpectrogram(n_mels=128)
        return transform(audio)

    def resample_audio(self, audio, sr):
        if sr != self.SR:
            print("resampling audio")
            resample = torchaudio.transforms.Resample(sr, self.SR)
            audio = resample(audio)
        return audio

    def pad_audio(self, audio):
        #slicing
        sample_len = int(self.SR*self.set_length)
        if audio.shape[0] > (sample_len) :
            audio = audio[:sample_len]
        # zero padding
        if audio.shape[0] < (sample_len) :
            pad_len = (sample_len) - audio.shape[0]
            pad_val = torch.zeros(pad_len)
            audio = torch.cat((audio, pad_val), dim=0)
        return audio

    def process_audio_file(self, path):
        audio, sr = torchaudio.load(path)
        audio = audio.squeeze(0)
        audio = self.resample_audio(audio, sr)
        audio = self.pad_audio(audio)
        if not self.raw_audio:
            mel_features = self.mel_transform(audio)
            mel_features = mel_features.unsqueeze(0) #adding 1 channel for cnn
            return mel_features
        else:
            return audio

    def __getitem__(self, item: int) :

        #pre-calculated mode
        audio_file = self.audio_files[item]
        if self.unlabeled_mode == False:
            return audio_file, self.targets[item]
        else:
            return audio_file, self.path_list[item]
            # return audio_file, self.file_names[item][:-4]