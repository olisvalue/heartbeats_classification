import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
import os, random, argparse


# custom
from models.CNNClassifier import Model
from models.PANClassifier import PANClassifier
from Trainer import Trainer
from MakeDataset import MakeDataset
from Evaluater import Evaluater
from CNNDataLoader import CNNDataLoader

# reproducibility
def initialization(seed = 0):   
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # multi-GPU
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 


parser = argparse.ArgumentParser(description='Settings for PrefixModel training')
parser.add_argument('-n', '--exp_name', type=str, default='no_name', help='name of the experiment')
parser.add_argument('-device', '--device', type = int, default=0, help='device number to use (if available)')
parser.add_argument('-train_bs', '--train_bs', type = int, default=20, help='train batch size')
parser.add_argument('-test_bs', '--test_bs', type = int, default=20, help='test batch size')

args = parser.parse_args()


total_epochs = 60
LR = 5e-5




#------------------Settings--------------------
#reproducibility
random_seed=42
initialization(seed=random_seed)
print("random_seed :", random_seed)

total_epochs = 55
LR = 7e-5

TEST_BATCH_SIZE = 30
TRAIN_BATCH_SIZE = 30

base_dir = '/data/valerii/heartbeats_classification/data/physionet'
train_dataloader = CNNDataLoader('train', TRAIN_BATCH_SIZE, base_dir, raw_audio=True)
test_dataloader = CNNDataLoader('test', TEST_BATCH_SIZE, base_dir, raw_audio=True,
                                weighted_sampler=False)

#============Experiment================
torch.cuda.empty_cache()

MODEL_NAME = args.exp_name 

USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda:'+ str(args.device) if USE_CUDA else 'cpu')
print("Using device: ", device)

model = PANClassifier(num_classes=2, device=args.device)

# params = torch.load('',\
#                      map_location='cuda:'+str(args.device))
# model.load_state_dict(params)

# print(args.device, type(args.device))
optimizer = AdamW(model.parameters(), lr=LR, weight_decay = 0.01)

trainer = Trainer(model, MODEL_NAME, train_dataloader, test_dataloader, optimizer, args.device, is_distributed=False)
trainer.train(total_epochs)


torch.cuda.empty_cache()
#============Experiment================