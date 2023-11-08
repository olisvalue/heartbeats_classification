import gradio as gr

import random
import torch
#from dataset_makers.util import *
from datahandlers.MyDataloader import *
from models.AAC_Prefix import *
import os


class Datasets:
    def __init__(self):
        self.hearts_dataset = MakeDataset(None, './data/WavCaps/'+ "hearts_kaggle", "test", None, 16000, tokens_size = None,tokenizer_type = None)
        self.caps_dataset = AudioCapsDataset(None, './data/AudioCaps', 'test', 26, tokens_size = 50,tokenizer_type = None)
        self.clotho_dataset = ClothoDataset(None, './data/Clotho', 'test', 26, tokens_size = 50,tokenizer_type = None)
        self.bbc_sounds = MakeDataset(None, './data/WavCaps/'+ "bbc_sounds", "test", None, 16000, tokens_size = None,tokenizer_type = None)
        self.soundbible = MakeDataset(None, './data/WavCaps/'+ "soundbible", "test", None, 16000, tokens_size = None,tokenizer_type = None)
    def select_dataset(self, dataset):
        if dataset == "audiocaps":
            self.dataset = self.caps_dataset
        elif dataset == "clotho":
            self.dataset = self.clotho_dataset
        elif dataset == "bbc_sounds":
            self.dataset = self.bbc_sounds
        elif dataset == "soundbible":
            self.dataset = self.soundbible
        elif dataset == "heartsounds":
            self.dataset = self.hearts_dataset
        else:
            self.dataset = None
        return self.dataset

class Model:
    def __init__(self, model_path, datasets, device = 'cpu'):
        self.device = device
        self.model = get_model_in_table(2, 2, device, weights_path = model_path,
                           prefix_size_dict = {"temporal_prefix_size" : 60, "global_prefix_size" : 20})
        # self.params = torch.load(model_path, weights_only=True)
        self.model.to(device)
        self.model.eval()
        self.datasets = datasets

    def random_predict(self, dataset):
        self.dataset = self.datasets.select_dataset(dataset)
        if self.dataset == None:
            return None, "Choose Dataset", ""
        audio_file, caption, path = random.choice(self.dataset)
        audio_file = audio_file.unsqueeze(0).to(self.device)
        predict = self.model(audio_file)[0][0]
        return path, caption, predict
    def predict_from_wav(self, audio_file):
        sr, audio = audio_file
        audio = torch.from_numpy(audio).unsqueeze(0)
        torchaudio.save("tmp.flac", audio, sample_rate=sr, format="flac")
        audio, sr = torchaudio.load("tmp.flac")
        audio = audio.to(self.device)
        predict = self.model(audio)[0][0]
        if os.path.exists("tmp.flac"):
            os.remove("tmp.flac")
        return predict
        
datasets = Datasets()
model = Model('/data/valerii/AudioCaption/data/old_model_recs/bbc_model_hearts2/best_model', datasets, device = 'cpu')

        


int_rand = gr.Interface(
    model.random_predict,
    [
        gr.Dropdown(
            ["audiocaps", "soundbible", "bbc_sounds", "clotho", "heartsounds"], label="Dataset", info="Choose test dataset"
        ),
    ],
    [
        gr.Audio(label = "Audio"),
        gr.Textbox(label = "Ground Truth Caption"),
        gr.Textbox(label = "Prediction"),
    ]
)


int_load = gr.Interface(
    model.predict_from_wav,
    [
        gr.Audio(label = "Audio", info = "Load your audio"),
    ],
    [
        gr.Textbox(label = "Prediction"),
    ]
)

demo = gr.TabbedInterface([int_rand, int_load], ["Predict from random test audio", "Predict from your audio"], title = "Audio captioning")
demo.launch(share=True)
