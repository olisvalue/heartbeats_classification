import torch
import torch.backends.cudnn as cudnn
import os
import sys
import random
import argparse

# custom
from dataset_makers.util import *
from transformers import GPT2Tokenizer
from AAC_Prefix.AAC_Prefix import * # network
from Trainer import *
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
parser.add_argument('-custom_tok', '--custom', action='store_true', help='use custom tokenizer')
parser.add_argument('-device', '--device', type = int, default=0, help='device number to use (if available)')
parser.add_argument('-ac', '--audiocaps', action = 'store_true', help='include audiocaps in training')
parser.add_argument('-cl', '--clotho', action = 'store_true', help='include clotho in training')
parser.add_argument('-as', '--audioset', action = 'store_true', help='include audioset in training')
parser.add_argument('-eac', '--eval_audiocaps', action = 'store_true', help='include audiocaps in test')
parser.add_argument('-ecl', '--eval_clotho', action = 'store_true', help='include clotho in test')
parser.add_argument('-eas', '--eval_audioset', action = 'store_true', help='include audioset in test')

args = parser.parse_args()
if args.custom:
    tokenizer = tokenizer_forCustomVocab(Dataset = 'AudioCaps')
    tokenizer_type = 'Custom'
    vocab_size = len(tokenizer.vocab)
else:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_type = 'GPT2'
    vocab_size = 50257

train_dataset_include = {
    'audiocaps': args.audiocaps,
    'clotho': args.clotho,
    'audioset': args.audioset
}
test_dataset_include = {
    'audiocaps': args.eval_audiocaps,
    'clotho': args.eval_clotho,
    'audioset': args.eval_audioset
}


total_epochs = 60
LR = 5e-5

temporal_prefix_size = 15
global_prefix_size = 11 
prefix_size = temporal_prefix_size + global_prefix_size

transformer_num_layers = {"temporal_num_layers" : 4, "global_num_layers" : 4}
prefix_size_dict = {"temporal_prefix_size" : temporal_prefix_size, "global_prefix_size" : global_prefix_size}


#------------------choose dataset--------------------

# control randomness
random_seed=2766
initialization(seed=random_seed)

print("random_seed :", random_seed)
print("vocab_size :", vocab_size)
    
TEST_BATCH_SIZE = 5
TRAIN_BATCH_SIZE = 10


test_dataloader  = MyDataLoader(test_dataset_include, 'test', TEST_BATCH_SIZE)
train_dataloader = MyDataLoader(train_dataset_include, 'train', TRAIN_BATCH_SIZE)


#============Experiment================
torch.cuda.empty_cache()

MODEL_NAME = args.exp_name
if tokenizer_type == 'Custom':
    MODEL_NAME += '_CustomHeader' 


USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda:'+ str(args.device) if USE_CUDA else 'cpu')
# device = torch.device('cuda' if USE_CUDA else 'cpu')

model = get_AAC_Prefix(tokenizer, tokenizer_type,
                        vocab_size = vocab_size, Dataset = 'AudioCaps',
                        prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers, 
                        encoder_freeze = False, decoder_freeze = True,
                        pretrain_fromAudioCaps = False, device = device)

# print(args.device, type(args.device)) s
warmup_steps = int((total_epochs * len(train_dataloader)) / 6)
num_training_steps=total_epochs * len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay = 0.01)
scheduler = get_cosine_schedule_with_warmup(
optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

trainer = Trainer(model, MODEL_NAME, train_dataloader, test_dataloader, optimizer, scheduler, args.device, is_distributed=False)
trainer.train(total_epochs)
# Train(model, LR, train_dataloader, test_dataloader,
#     epochs, model_name = MODEL_NAME, beam_search = True, device = device,
#     Dataset = 'AudioCaps')

torch.cuda.empty_cache()
#============Experiment================