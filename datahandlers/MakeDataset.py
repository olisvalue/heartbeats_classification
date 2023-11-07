import torch
from torch.utils.data import Dataset

import pandas as pd
import torchaudio
import os
from tqdm import tqdm
import re

def caption_preprocess(caption):
    caption = caption.lower()    
    caption = caption.replace(',', ' , ') 
    caption = re.sub(' +', ' ', caption)
    caption = caption.replace(' ,', ',')
    caption = re.sub(r'[.]', '', caption)
    caption += '.'
    caption = caption.strip()
    return caption

def convert_to_list(input_data):
    if isinstance(input_data, str):
        return [input_data]
    elif isinstance(input_data, list):
        return input_data
    else:
        raise ValueError("Caption must be a string or a list")

class MakeDataset(Dataset):
    def __init__(self, tokenizer, base_dir, split, prefix_size, SR, tokens_size = 30, tokenizer_type = 'GPT2') :  # split = 'train' or 'test'
        super(MakeDataset, self).__init__()
        
        self.SAMPLE_RATE = SR
        # self.resample = torchaudio.transforms.Resample(audio_sr, self.SAMPLE_RATE)

        self.split = split

        self.data_dir = base_dir + '/' + split + '/'
        json_df = pd.read_json(base_dir + '/' + 'dataset.json')
        json_df = json_df["data"]

        # file's name = youtube_id
        #audio_file_list = os.listdir(self.data_dir)
        
        self.path_list = []
        self.token_list = []
        self.caption_list_for_test = []

        self.file_names = set(os.listdir(self.data_dir))
        # self.audio_files = []
        
        for i in tqdm(range(len(json_df)), desc = f'get {split} dataset from {base_dir}...'):
            file_id = json_df[i]["id"]
            captions = json_df[i]["caption"]
            captions = convert_to_list(captions)
            file_name = file_id + ".flac"
            if file_name not in self.file_names:
                continue
            
            self.path_list.append(self.data_dir+file_name)
            
            # self.add_audio_file(self.data_dir + self.path_list[-1])

            for caption in captions : 
                caption = caption_preprocess(caption)
                if split != 'train' :
                    self.caption_list_for_test.append(caption)
                elif split == 'train' :
                    if tokenizer_type == 'GPT2' :
                        tokens = tokenizer(caption)['input_ids']
                    else :
                        tokens = tokenizer.encode(caption)
                    self.token_list.append(torch.tensor(tokens))
    

        if split == 'train' :          
            # self.all_len = torch.tensor([len(self.token_list[i]) for i in range(len(self.token_list))]).float()
            # self.max_seq_len = min(int(self.all_len.mean() + self.all_len.std() * 4), int(self.all_len.max()))
            # print("95%+ max tok len:", int(self.all_len.mean() + self.all_len.std() * 4))
            # print("max tok len:", int(self.all_len.max()))
            self.max_seq_len = tokens_size
        self.prefix_length = prefix_size # audio_prefix_length + semantic_prefix_length
            
    def __len__(self):
        return len(self.path_list)
    
    def add_audio_file(self, path):
        audio_file, sr = torchaudio.load(path)
        audio_file = audio_file.squeeze(0)
        if sr != self.SAMPLE_RATE:
            print("resampling audio")
            resample = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            audio_file = resample(audio_file)
         # slicing or padding based on set_length
        # slicing
        # if audio_file.shape[0] > (self.SAMPLE_RATE * self.set_length) :
        #     audio_file = audio_file[:self.SAMPLE_RATE * self.set_length]
        # # zero padding
        # if audio_file.shape[0] < (self.SAMPLE_RATE * self.set_length) :
        #     pad_len = (self.SAMPLE_RATE * self.set_length) - audio_file.shape[0]
        #     pad_val = torch.zeros(pad_len)
        #     audio_file = torch.cat((audio_file, pad_val), dim=0)

        #or compression
        #audio_file = self.compress_audio(audio_file)

        self.audio_files.append(audio_file)

    def pad_tokens(self, item: int):
        tokens = self.token_list[item].clone().detach()
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.token_list[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.token_list[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask
    
    def compress_audio(self, audio, set_length = 10) :
        ratio = audio.size()[0]/(self.SAMPLE_RATE * set_length)
        compress_idx_list = []
        for idx in range(self.SAMPLE_RATE * set_length) :
            compress_idx_list.append(int(ratio * idx))

        return audio[compress_idx_list]
    def get_all_len(self):
        return self.all_len
    
    
    def __getitem__(self, item: int) :
        audio_file, sr = torchaudio.load(self.path_list[item])
        audio_file = audio_file.squeeze(0)
        if sr != self.SAMPLE_RATE:
            print("resampling audio")
            resample = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            audio_file = resample(audio_file)

        if self.split == 'train' :
            tokens, mask = self.pad_tokens(item)
            return audio_file, tokens, mask, self.path_list[item]
        else :
            return audio_file, self.caption_list_for_test[item], self.path_list[item]