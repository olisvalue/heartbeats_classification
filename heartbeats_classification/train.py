import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
import os, random, argparse


# custom
from models.CNNClassifier import Model
from Trainer import Trainer
from MakeDataset import MakeDataset
from Evaluater import Evaluater

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


TEST_BATCH_SIZE = 60
TRAIN_BATCH_SIZE = 60



base_dir = '/home/stud_valery/simple_audio_classification/data'
train_dataset = MakeDataset(base_dir, 'train')
test_dataset = MakeDataset(base_dir, 'test', unlabeled_mode=True)



train_dataloader = DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True, num_workers=8, drop_last=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=TEST_BATCH_SIZE,
                              shuffle=False, num_workers=8, drop_last=False)

#============Experiment================
torch.cuda.empty_cache()

MODEL_NAME = args.exp_name 

USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda:'+ str(args.device) if USE_CUDA else 'cpu')
print("Using device: ", device)

model = Model(num_classes=2, device=args.device)

# params = torch.load('',\
#                      map_location='cuda:'+str(args.device))
# model.load_state_dict(params)

# print(args.device, type(args.device))
optimizer = AdamW(model.parameters(), lr=LR, weight_decay = 0.01)

trainer = Trainer(model, MODEL_NAME, train_dataloader, test_dataloader, optimizer, args.device, is_distributed=False, without_eval=True)
trainer.train(total_epochs)


torch.cuda.empty_cache()
#============Experiment================