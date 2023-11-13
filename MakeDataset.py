import pandas as pd
import torchaudio, torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

from collections import defaultdict

class MakeDataset(Dataset):
    def __init__(self, base_dir, split, SR=16000, set_length = 35, raw_audio=False,
                unlabeled_mode=False, filename_mode=False, zero_paddings=True,
                llm_mode=False, prefix_length=-1) :  # split = 'train' or 'test'
        super(MakeDataset, self).__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_length = prefix_length
        self.llm_mode = llm_mode
        self.zero_paddings = zero_paddings
        self.raw_audio=raw_audio
        self.unlabeled_mode=unlabeled_mode
        self.split = split
        self.set_length = set_length
        self.SR = SR

        self.data_dir = base_dir + '/' + split + '/'
        df = pd.read_csv(base_dir + '/dataset.csv')
        word_to_number = {"Normal": 0, "normal": 0,
                          "Abnormal": 1, "artifact": 1, "murmur": 1, "extrahls": 1, "extrastole" : 1, "rubs":1}
        df['label'] = df['label'].map(word_to_number)
        label_dict = df.set_index('id').to_dict()['label']
        if self.llm_mode == True:
            captions_dict = df.set_index('id').to_dict()['caption']

        self.path_list = []
        self.targets = []
        self.captions = []
        self.tokens = []
        self.max_tok_len = 40
        self.masks = []
        self.file_names = list(set(os.listdir(self.data_dir)))

        self.audio_files = []

        for file_name in tqdm(self.file_names, desc = f'get {split} dataset from {base_dir}...'):
            self.path_list.append(self.data_dir+file_name)
            if self.unlabeled_mode==False:
                key = file_name[:-5] if filename_mode==False else file_name
                # if df_dict[key] == 0:
                #     continue
                self.targets.append(label_dict[key])
            if self.llm_mode==True:
                caption = captions_dict[key]
                self.tokens.append(self.tokenizer(caption)['input_ids'])
        
            self.audio_files.append(self.process_audio_file(self.path_list[-1]))
        
        if self.llm_mode==True:
            self.postprocess_all_tokens()
        

    def __len__(self):
        return len(self.audio_files)
    
    def postprocess_all_tokens(self):
        for i in range(len(self.audio_files)):
            self.postprocess_item_tokens(i)

    def postprocess_item_tokens(self, item: int):
        tokens = torch.tensor(self.tokens[item])
        padding = self.max_tok_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_tok_len]
            self.tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        self.masks.append(mask)
        return tokens, mask

    def mel_transform(self, audio):
        transform = torchaudio.transforms.MelSpectrogram(n_mels=128)
        return transform(audio)

    def resample_audio(self, audio, sr):
        if sr != self.SR:
            print("resampling audio")
            resample = torchaudio.transforms.Resample(sr, self.SR)
            audio = resample(audio)
        return audio
    
    def periodic_pad_audio(self, audio):
        #slicing
        sample_len = int(self.SR*self.set_length)
        if audio.shape[0] > (sample_len) :
            audio = audio[:sample_len]
        # zero padding
        if audio.shape[0] < (sample_len) :
            audio = torch.cat((audio, audio), dim=0)
            audio = self.periodic_pad_audio(audio)
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
        if self.zero_paddings == True:
            audio = self.pad_audio(audio)
        else:
            audio = self.periodic_pad_audio(audio)
        if not self.raw_audio:
            mel_features = self.mel_transform(audio)
            mel_features = mel_features.unsqueeze(0) #adding 1 channel for cnn
            return mel_features
        else:
            return audio

    def __getitem__(self, item: int) :

        #pre-calculated mode
        audio_file = self.audio_files[item]

        if self.llm_mode == True:
            return audio_file, self.targets[item], self.tokens[item], self.masks[item] 

        elif self.unlabeled_mode == False:
            return audio_file, self.targets[item]
        else:
            return audio_file, self.path_list[item]
            # return audio_file, self.file_names[item][:-4]