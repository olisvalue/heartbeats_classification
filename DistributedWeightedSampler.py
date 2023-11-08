import torch.distributed as dist
import torch
from torch.utils.data import Sampler
import math

class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, total_num_samples, is_shuffle, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.weights = weights
        self.total_num_samples = total_num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.local_num_samples = int(math.ceil(total_num_samples * 1.0 / self.num_replicas))
        self.total_size = self.local_num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = is_shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(self.total_num_samples, generator=g).tolist()
        else:
            indices = list(range(self.total_num_samples))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.local_num_samples


        return iter(torch.multinomial(self.weights[indices], self.local_num_samples, self.replacement).tolist())

    def __len__(self):
        return self.local_num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch