## Proposed model

This repository contains a project using the CNN+LLM scheme to improve the classification of heartbeat sounds.   
The physionet 2016 dataset was used as the main data for training and validation: https://archive.physionet.org/challenge/2016/.

The idea of this project is inspired by the following paper: https://arxiv.org/abs/2303.17489

Here is schematic diagram of the proposed pipeline:

![Alt text](<./assets/pipeline.png>)


## Achieved results.
0.904 UAR with CNN(PANNs) only, as in the following paper: https://ieeexplore.ieee.org/document/9175450.   
0.941 UAR with proposed pipeline.


## Usage and reproduction.

Here are the steps you should follow to reproduce the results:
1. Obtain heartbeats data from physionet 2016 challenge (https://archive.physionet.org/challenge/2016/)
2. Do preprocessing of the data (1-channel 16khz audio, train and test directories in ./data/physionet)
2. Use `python llm_parallel_train.py -n "your_exp_name"` to train the model. Note, that you should choose number of gpus for training in this line: `world_size = 4`, and select them in this line: `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"`. 


Check `./models/PrefixLLM.py` to take a look at the implementation of the model.   
Check `./datahandlers/MakeDataset.py` and `./datahandlers/CNNDataLoader.py` to adjust data processing if necessary.   
Check `./llm_classification/LLMTrainer.py` to setup experiment settings.    
Check `pipeline.ipynb` to find evaluation tools and more info about the project.    
