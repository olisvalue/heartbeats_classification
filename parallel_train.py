import torch
import os
import sys
import random
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

# custom
from CNNDataLoader import CNNDataLoader
from models.PANClassifier import * # network
from Trainer import *

def ddp_setup(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12417"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        torch.multiprocessing.set_sharing_strategy('file_system')

        print("my rank: ", rank, " world size: ", world_size)
        
def main(rank: int, world_size: int, total_epochs: int, model_name: str):
        ddp_setup(rank, world_size)
        # try:  
        model = PANClassifier(num_classes=2, device=rank)

        weights_path = '/home/stud_valery/heartbeats_classification/data/train_record/pann_balanced1/best_model'
        params = torch.load(weights_path, map_location='cuda:' + str(rank))
        model.load_state_dict(params)
        TRAIN_BATCH_SIZE = 30
        TEST_BATCH_SIZE = 30

        base_dir = '/home/stud_valery/heartbeats_classification/data/physionet'
        train_dataloader = CNNDataLoader('train', TRAIN_BATCH_SIZE, base_dir, raw_audio=True,
                                            weighted_sampler=True, is_distributed=True)
        test_dataloader = CNNDataLoader('test', TEST_BATCH_SIZE, base_dir, raw_audio=True,
                                        weighted_sampler=False, is_distributed=True)


        optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay = 0.01)
        trainer = Trainer(model, model_name, train_dataloader, test_dataloader, optimizer, rank, is_distributed=True)
        trainer.train(total_epochs, start_epoch=53)
        destroy_process_group()
        # finally:
        #     print("Smth went wrong with DDP")
        #     destroy_process_group()
if __name__ == "__main__":
    world_size = 2
    mp.spawn(main, args=(world_size, 50, "pann_balanced1"), nprocs=world_size)