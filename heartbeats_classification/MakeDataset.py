import pandas as pd
import torchaudio, torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os

class MakeDataset(Dataset):
    def __init__(self, base_dir, split, SR=16000, set_length = 10.3, unlabeled_mode=False) :  # split = 'train' or 'test'
        super(MakeDataset, self).__init__()

        self.unlabeled_mode=unlabeled_mode
        self.split = split
        self.set_length = set_length
        self.SR = SR

        self.data_dir = base_dir + '/' + split + '/'
        df = pd.read_csv(base_dir + '/targets.tsv', sep='\t', header=None, names=['filename', 'class'])
        df_dict = df.set_index('filename').to_dict()['class']

        self.path_list = []
        self.targets = []
        self.file_names = list(set(os.listdir(self.data_dir)))[:100]

        self.audio_files = []

        for file_name in tqdm(self.file_names, desc = f'get {split} dataset from {base_dir}...'):
            self.path_list.append(self.data_dir+file_name)
            if self.unlabeled_mode==False:
                self.targets.append(df_dict[file_name[:-4]])
            self.audio_files.append(self.process_audio_file(self.path_list[-1]))
        
    def __len__(self):
        return len(self.path_list)
    
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
        #audio_file = resample_audio(audio_file)
        audio = self.pad_audio(audio)
        mel_features = self.mel_transform(audio)
        mel_features = mel_features.unsqueeze(0) #adding 1 channel for cnn
        return mel_features

    def __getitem__(self, item: int) :
        #pre-calculated mode
        audio_file = self.audio_files[item]
        if self.unlabeled_mode == False:
            return audio_file, self.targets[item]
        else:
            return audio_file, self.path_list[item]
            # return audio_file, self.file_names[item][:-4]