import pickle
import re
import csv
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Tokenizer

from datahandlers.AudioCapsDataset import *
from datahandlers.ClothoDataset import *
from datahandlers.MakeDataset import *

from torch.utils.data import Sampler
import torch.distributed as dist

class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        start_indice = int(self.total_size * self.rank/self.num_replicas)
        end_indice = int(self.total_size * (self.rank+1)/self.num_replicas)
        if start_indice % 5 != 0:
            start_indice = start_indice - start_indice % 5
        if end_indice % 5 != 0:
            end_indice = end_indice - end_indice % 5  
        self.indices = indices[start_indice:end_indice]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = self.indices
        # assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch

# Tokenizer of own vocabulary for Training Datset
class tokenizer_forCustomVocab() :
    
    def encode(self, sentence) :
        
        word_list = sentence.split(' ')
        
        token_idx = []
        for word in word_list : 
            token_idx.append(self.vocab.index(word))
            
        #  <eos> 
        token_idx.append(13)
        
        return token_idx
    
    def decode(self, token_idx) :
        
        sentence = ''
        for idx in token_idx :
            if (idx == 13) :
                break
            else :
                sentence += self.vocab[idx] + ' '

        sentence = sentence.rstrip()
        sentence = re.sub(r'[.]', '', sentence)

        sentence += '.'
        
        return sentence

    def __init__(self, Dataset) : # Dataset = 'AudioCaps' or 'Clotho'
        
        file_path = ''

        if Dataset == 'AudioCaps' :
            file_path = './data/AudioCaps/AudioCaps_vocabulary.pickle'
            with open(file_path, 'rb') as f:
                self.vocab = pickle.load(f) 
        elif Dataset == 'Clotho' :
            file_path = './data/Clotho/Clotho_vocabulary.pickle'
            with open(file_path, 'rb') as f:
                self.vocab = pickle.load(f) 


class MyDataLoader(DataLoader):
    def __init__(self, dataset_include, mode, batch_size, is_distributed = False, tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                 , tokenizer_type = 'GPT2', tokens_size = 40, prefix_size = 26, SR = 16000, nb_references = 5):
        self.train = True if mode == 'train' else False
        self.SR = SR
        self.max_len_train = 160000#SR * 31.25
        self.max_len_test = 160000#SR * 60
        self.nb_references = nb_references
        collate_fn = self.collate_fn_train if self.train else self.collate_fn_test

        if self.train:
            is_shuffle = True
            is_drop_last = True
        else :
            is_shuffle = False
            is_drop_last = False
        
        datasets = []

        if 'audiocaps' in dataset_include:
            if dataset_include['audiocaps']:
                datasets.append(AudioCapsDataset(tokenizer, './data/AudioCaps', mode, prefix_size, tokens_size = tokens_size,tokenizer_type = tokenizer_type))
            dataset_include.pop('audiocaps')
        if 'clotho' in dataset_include:
            if dataset_include['clotho']:
                datasets.append(ClothoDataset(tokenizer, './data/Clotho', mode, prefix_size, tokens_size = tokens_size,tokenizer_type = tokenizer_type))
            dataset_include.pop('clotho')
        for name in dataset_include:
            if dataset_include[name]:
                datasets.append(MakeDataset(tokenizer, './data/WavCaps/'+ name, mode, prefix_size, self.SR, tokens_size = tokens_size,tokenizer_type = tokenizer_type))
        
        concatenated_dataset = ConcatDataset(datasets)

        if is_distributed:
            cpu_core_num = 0
            sampler = DistributedSampler(concatenated_dataset, shuffle = is_shuffle) if self.train else DistributedEvalSampler(concatenated_dataset)
            is_shuffle = False
        else:
            cpu_core_num = 8
            sampler = None
        super(MyDataLoader, self).__init__(dataset=concatenated_dataset,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=cpu_core_num,
                      drop_last=is_drop_last, collate_fn=collate_fn,
                      sampler = sampler)
        
    def collate_fn_train(self, batch):
            list_of_tensors = [item[0] for item in batch]
            ###dynamic length
            # max_len = max([tensor.shape[0] for tensor in list_of_tensors])
            ###static length
            max_len = self.max_len_train
            if max_len > self.max_len_train:
                max_len = self.max_len_train
            for i, tensor in enumerate(list_of_tensors):
                pad_len = max_len - tensor.shape[0]
                if pad_len < 0:
                    pad_tensor = tensor[:max_len]
                else:
                    pad_val = torch.zeros(pad_len)
                    pad_tensor = torch.cat((tensor, pad_val), dim=0)
                list_of_tensors[i] = pad_tensor
            
            audio_tensor = torch.stack(list_of_tensors)
            tensors = [torch.stack([item[i] for item in batch]) for i in range(1, 3)]
            file_names = [item[3] for item in batch]
            return [audio_tensor] + tensors + [file_names]
    def collate_fn_test(self, batch):
        list_of_tensors = [item[0] for item in batch]
        max_len = self.max_len_test
        for i, tensor in enumerate(list_of_tensors):
            pad_len = max_len - tensor.shape[0]
            if pad_len < 0:
                pad_tensor = tensor[:max_len]
            else:
                pad_val = torch.zeros(pad_len)
                pad_tensor = torch.cat((tensor, pad_val), dim=0)
            list_of_tensors[i] = pad_tensor

        caps_filenames = [[item[i] for item in batch] for i in range(1, 3)]
        return [list_of_tensors] + caps_filenames


def get_pred_captions(model, test_dataloader, device, dataset = 'AudioCaps') :
    model.eval()
    
    with open(f"{dataset}_pred_captions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'caption'])
        for i, (audio, captions, f_names) in enumerate(tqdm(test_dataloader, desc="Get Caption...")):
            with torch.no_grad() :
                audio = audio.to(device)
                audio = audio[0,:].unsqueeze(0)

                pred_caption = model(audio, None, beam_search = True)[0][0]

                writer.writerow([f_names[0], pred_caption])