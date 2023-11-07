import torch
from torch.utils.data import Dataset

import pandas as pd
import torchaudio
import os
from tqdm import tqdm
import re
import string

class ClothoDataset(Dataset):
    def compress_audio(self, audio, set_length = 10) :
        
        ratio = audio.size()[1]/(self.SAMPLE_RATE * set_length)
        
        compress_idx_list = []
        
        for idx in range(self.SAMPLE_RATE * set_length) :
            compress_idx_list.append(int(ratio * idx))
        
        return audio[:, compress_idx_list]
    
    def __init__(self, tokenizer, data_dir, split, prefix_size, tokens_size = 30, tokenizer_type = 'GPT2') :  # split = 'train' or 'test'
        super(ClothoDataset, self).__init__()
        
        self.SAMPLE_RATE = 16000
        
        self.change_sampling_rate = torchaudio.transforms.Resample(self.SAMPLE_RATE, 16000)
        
        self.split = split
        
        self.audio_files_dir = data_dir + '/clotho_audio_files/' + split
        
        csv_file_path = data_dir + '/clotho_csv_files/' + 'clotho_captions_' + split + '.csv'
        
        audio_file_list = os.listdir(self.audio_files_dir)
        # self.audio_file_list = []
        self.token_list = []
        self.caption_list_for_test = []

        self.pad_size = 500000
        
        self.audio_path_list = []

        csv_file = pd.read_csv(csv_file_path)
        
        for file in tqdm(audio_file_list, desc = 'get dataset...') :

            audio_file_full_path = self.audio_files_dir + '/' + file 
            # audio_file, _ = torchaudio.load(audio_file_full_path)
            # audio_file = audio_file.squeeze(0)
            
            # if is_settingnum_3 == False :
            #     audio_file = audio_file.squeeze(0)
            #     # slicing or padding based on set_length
            #     set_length = 30

            # slicing or padding based on set_length
            # slicing
            # if audio_file.shape[0] > (self.pad_size) :
            #     audio_file = audio_file[:self.pad_size]
            # # zero padding
            # if audio_file.shape[0] < (self.pad_size) :
            #     pad_len = (self.pad_size) - audio_file.shape[0]
            #     pad_val = torch.zeros(pad_len)
            #     audio_file = torch.cat((audio_file, pad_val), dim=0)
            
            for i in range(5) :
                self.audio_path_list.append(audio_file_full_path)
                # self.audio_file_list.append(audio_file)
                sentence_str = 'caption_' + str(i + 1)
                caption = csv_file[csv_file['file_name'] == file][sentence_str].item()

                caption = caption.lower()    
                
                caption = caption.replace(',', ' , ') 
                caption = re.sub(' +', ' ', caption)
                caption = caption.replace(' ,', ',')
                caption = re.sub(r'[.]', '', caption)
                
                caption = caption.strip()
                caption += '.'
                
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
            self.prefix_length = prefix_size
            self.mask_list = self.pad_token_list()
            
    def __len__(self):
       
        return len(self.audio_path_list)
    
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
        audio_file, _ = torchaudio.load(self.audio_path_list[item])
        audio_file = audio_file.squeeze(0)

        if self.split == 'train' : 
            return audio_file, self.token_list[item], self.mask_list[item], self.audio_path_list[item]
        else :
            return audio_file, self.caption_list_for_test[item], self.audio_path_list[item]