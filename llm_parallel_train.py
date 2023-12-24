import torch
import os
import sys
import random
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

# custom
from datahandlers.CNNDataLoader import CNNDataLoader
from models.PrefixLLM import * # network
from llm_classification.LLMTrainer import *

def ddp_setup(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12417"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        torch.multiprocessing.set_sharing_strategy('file_system')

        print("my rank: ", rank, " world size: ", world_size)
        
def main(rank: int, world_size: int, total_epochs: int, model_name: str):
        ddp_setup(rank, world_size)
        try:  
            prefix_length = 80
            prefix_size_dict = {"temporal_prefix_size" : 60, "global_prefix_size" : 20}
            model = PrefixLLM(encoder_freeze=False, 
                              prefix_size_dict = prefix_size_dict, device = rank,
                              llm_mode=True)
            # weights_path = './models/resources/bbc_model2.pt'
            # params = torch.load(weights_path, map_location='cuda:' + str(rank))
            # model.load_state_dict(params, strict=False)

            # model.activate_audio_encoder()

            base_dir = './data/physionet'

            TRAIN_BATCH_SIZE = 18
            TEST_BATCH_SIZE = 18

            train_dataloader = CNNDataLoader('train', TRAIN_BATCH_SIZE, base_dir, raw_audio=True, weighted_sampler=True, 
                                             is_distributed=True, llm_mode=True, prefix_length=prefix_length)
            test_dataloader = CNNDataLoader('test', TEST_BATCH_SIZE, base_dir, raw_audio=True, weighted_sampler=False, 
                                        is_distributed=True, llm_mode=False, prefix_length=prefix_length)

            warmup_steps = int((total_epochs * len(train_dataloader)) / 6)
            num_training_steps=total_epochs * len(train_dataloader)
            optimizer = AdamW(model.parameters(), lr=4e-3, weight_decay = 0.01)
            scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

            trainer = LLMTrainer(model, model_name, train_dataloader, test_dataloader, optimizer, scheduler, rank, 
                                 is_distributed=True, accum_iter=1)

            print("trainer initialized")

            trainer.train(total_epochs)
        finally:
            destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(main, args=(world_size, 150, "rawllm_audiosetpanns"), nprocs=world_size)