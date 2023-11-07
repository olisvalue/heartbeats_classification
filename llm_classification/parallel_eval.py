import torch
import os
import sys
import random
from transformers import GPT2Tokenizer
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

# custom
from datahandlers.MyDataloader import *
from models.SpeechEncCaption import * # network
from Trainer import *
    
TEST_BATCH_SIZE = 5

# USE_CUDA = torch.cuda.is_available() 
# device = torch.device('cuda' if USE_CUDA else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer_type = 'GPT2'
def ddp_setup(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12417"
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        # os.environ["OMP_NUM_THREADS"] = "1" 
        # os.environ["MKL_NUM_THREADS"] = "1" 
        # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        torch.multiprocessing.set_sharing_strategy('file_system')

        print("my rank: ", rank, " world size: ", world_size)
        
def main(rank: int, world_size: int, total_epochs: int, model_name: str):
        ddp_setup(rank, world_size)
        try:
            prefix_length = 60
            model = SpeechEncCaption(encoder_freeze=True, map_networks_freeze=False,
                                  decoder_freeze=True,
                                  prefix_length=prefix_length,
                                  use_mapping_network=True,
                                  use_cnn=False,
                                  device = rank)
            # model = get_AAC_Prefix(GPT2Tokenizer.from_pretrained("gpt2"), 'GPT2',
            #                 vocab_size = 50257, Dataset = 'AudioCaps',
            #                 prefix_size_dict = {"temporal_prefix_size" : 15, "global_prefix_size" : 11}, transformer_num_layers = {"temporal_num_layers" : 4, "global_num_layers" : 4}, 
            #                 encoder_freeze = False, decoder_freeze = True,
            #                 pretrain_fromAudioCaps = False, device = rank)
            params = torch.load('/data/valerii/AudioCaption/data/Train_record/hubert_audiocaps4/best_model', map_location='cuda:'+str(rank))
            model.load_state_dict(params)
    
            # train_dataloader = MyDataLoader({}, 'train', 20)
            # train_dataloader = MyDataLoader({"audiocaps": True, "audioset": True, "clotho": True, "bbc_sounds": True}, 'train', 15, is_distributed=True)
            # test_dataloader = MyDataLoader({"bbc_sounds": True, "clotho": True, "soundbible": True, "audiocaps": True}, 'test', 15, is_distributed=True)
            test_dataloader = MyDataLoader({"audiocaps": True},
                                            'test', 20, is_distributed=True, nb_references=5)
            # warmup_steps = int((total_epochs * len(train_dataloader)) / 6)
            # num_training_steps=total_epochs * len(train_dataloader)
            # optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay = 0.01)
            # scheduler = get_cosine_schedule_with_warmup(
            # optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
            
            # print("starting initialization of trainer")
            trainer = Trainer(model, model_name, None, test_dataloader, None, None, rank, is_distributed=True)

            print("trainer initialized")

            trainer.evaluate()
        finally:
            destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(main, args=(world_size, 55, "as_ac_cl_bbc_const600k_3"), nprocs=world_size)