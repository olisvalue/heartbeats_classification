import torch
import torch.nn as nn
from models.PANNs.CNN14 import Cnn14


class PANClassifier(nn.Module):
    def __init__(self, num_classes, device, LLM_mode=False):
        super(PANClassifier, self).__init__()
        self.LLM_mode = LLM_mode
        self.cnn = Cnn14(sample_rate=16000, window_size=512, 
                hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                classes_num=527)
        checkpoint_path = "./models/PANNs/Cnn14_16k_mAP=0.438.pth"
        if type(device) == int:
            checkpoint = torch.load(checkpoint_path, map_location='cuda:'+str(device))
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        self.cnn.load_state_dict(checkpoint['model'])
        
        self.cnn.fc_audioset = nn.Linear(in_features=2048, out_features=2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        temp, glob = self.cnn(x)
        x = self.softmax(glob)

        if self.training:
            if self.LLM_mode==True:
                return temp, glob
            else:
                return x
        else:
            return x.argmax(dim=1).to('cpu')