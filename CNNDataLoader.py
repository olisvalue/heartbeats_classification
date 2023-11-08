from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from MakeDataset import MakeDataset

from torch.utils.data import Sampler
import torch.distributed as dist
import numpy as np
import torch


from DistributedWeightedSampler import DistributedWeightedSampler
from DistributedEvalSampler import DistributedEvalSampler

class CNNDataLoader(DataLoader):
    def __init__(self, mode, batch_size, base_dir, weighted_sampler=True, raw_audio=False, is_distributed=False, SR=16000):

        self.raw_audio = raw_audio
        self.base_dir = base_dir
        self.train = True if mode == 'train' else False
        self.SR = SR
        self.len_train = 160000 #SR * 31.25
        self.len_test = 160000

        dataset = MakeDataset(self.base_dir, mode, raw_audio=self.raw_audio)
        sampler = None

        if self.train:
            is_drop_last = True
            is_shuffle = True
            if weighted_sampler:
                samples_weight = self.get_samples_weight(dataset)
                if is_distributed:
                    sampler = DistributedWeightedSampler(samples_weight, len(dataset),
                                                         is_shuffle=is_shuffle, replacement=True)
                else:
                    sampler = WeightedRandomSampler(samples_weight, len(dataset), replacement=True)
                is_shuffle=False
        else :
            is_shuffle = False
            is_drop_last = False
        
        cpu_core_num = 8
        if sampler is None:
            if is_distributed:
                cpu_core_num = 0
                sampler = DistributedSampler(dataset, shuffle = is_shuffle) if self.train else DistributedEvalSampler(dataset)
                is_shuffle = False
            else:
                sampler = None  

        super(CNNDataLoader, self).__init__(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=cpu_core_num,
                      drop_last=is_drop_last,
                      sampler = sampler)
        
    def get_samples_weight(self, dataset):
        y_train = [dataset.targets[i] for i in range(len(dataset))]
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = torch.tensor([weight[t] for t in y_train])
        return samples_weight