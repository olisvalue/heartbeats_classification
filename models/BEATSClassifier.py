import torch
import torch.nn as nn
from models.PANNs.CNN14 import Cnn14

from models.efficientat.dymn.model import get_model as get_dymn
class BEATSClassifier(nn.Module):
    def __init__(self, num_classes, llm_mode=False):
        super(BEATSClassifier, self).__init__()
        self.llm_mode = llm_mode
        self.cnn = get_dymn(pretrained_name="dymn20_as", width_mult = 2.0)

        self.cnn.classifier[5] = nn.Linear(in_features=2560, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        logits, embeds, temp_f = self.cnn(x)
        x = self.softmax(logits)

        if self.llm_mode:
            return temp_f, embeds
        else:
            if self.training:
                return x
            else:
                return x.argmax(dim=1).to('cpu')