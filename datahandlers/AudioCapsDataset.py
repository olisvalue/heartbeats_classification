import torch
from torch.utils.data import Dataset

import pandas as pd
import torchaudio
import os
from tqdm import tqdm
import re

class AudioCapsDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, prefix_size, tokens_size = 30, tokenizer_type = 'GPT2') :  # split = 'train' or 'test'
        super(AudioCapsDataset, self).__init__()

        self.SAMPLE_RATE = 16000
        self.split = split
        
        self.data_dir = data_dir + '/' + split + '/'
        csv_file = pd.read_csv(self.data_dir + split + '.csv')
        
        # file's name = youtube_id
        audio_file_list = os.listdir(self.data_dir)
        
        self.path_list = []
        self.token_list = []
        self.caption_list_for_test = []

        self.audio_files = []
        self.pad_size = 500000
        
        for file in tqdm(audio_file_list, desc = 'get dataset...') :
            if file[-3:] == 'wav' :
                file_row_in_csv = csv_file[csv_file['youtube_id'] == file[:-4]]
                
                captions = file_row_in_csv['caption'].to_list() # train : 1 caption per each audio, test : 5 captions per each audio
                for caption in captions :
                    self.path_list.append(self.data_dir + file)
                    # audio_file, _ = torchaudio.load(self.data_dir+file)
                    # audio_file = audio_file.squeeze(0)

                    # slicing or padding based on set_length
                    # slicing
                    # if audio_file.shape[0] > (self.pad_size) :
                    #     audio_file = audio_file[:self.pad_size]
                    # # zero padding
                    # if audio_file.shape[0] < (self.pad_size) :
                    #     pad_len = (self.pad_size) - audio_file.shape[0]
                    #     pad_val = torch.zeros(pad_len)
                    #     audio_file = torch.cat((audio_file, pad_val), dim=0)

                    # self.audio_files.append(audio_file)
                    
                    caption = self.caption_preprocess(caption)
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
        self.mask_list = self.pad_token_list()

    def __len__(self):
       
        return len(self.path_list)
    
    def caption_preprocess(self, caption):
        caption = caption.lower()    
        caption = caption.replace(',', ' , ') 
        caption = re.sub(' +', ' ', caption)
        caption = caption.replace(' ,', ',')
        caption = re.sub(r'[.]', '', caption)
        caption += '.'
        caption = caption.strip()
        return caption

    def pad_token_list(self):
        mask_list = []
        for i in range(len(self.token_list)): 
            tokens, mask = self.pad_token(i)
            self.token_list[i] = tokens
            mask_list.append(mask)
        return mask_list
    def pad_token(self, item: int):
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
        
    # def get_all_len(self):
    #     return self.all_len
    
    
    def __getitem__(self, item: int) :
        audio_file, _ = torchaudio.load(self.path_list[item])
        audio_file = audio_file.squeeze(0)
        
        if self.split == 'train' :
            return audio_file, self.token_list[item], self.mask_list[item], self.path_list[item]
        else :
            return audio_file, self.caption_list_for_test[item], self.path_list[item]