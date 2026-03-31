import torch
import random
import numpy as np




class DataLoader :
    def __init__(
        self, 
        datapath: str,
        batch_size: int,
        context_len: int,
        device: torch.device
    ):
        self.batch_size = batch_size
        self.context_len = context_len
        self.device = device
        self.dataset = np.load(datapath, mmap_mode="r")
        self.valid_idx = len(self.dataset) - context_len - 1

    def __iter__(self) :
        return self

    def __next__(self) :

        # start_indices = [random.randint(0, valid_idx) for _ in range(batch_size)]
        start_indices = random.choices(range(self.valid_idx + 1), k= self.batch_size)

        token_ids_list = [self.dataset[start_idx : start_idx + self.context_len + 1] for start_idx in start_indices]
        x_token_ids = [token_ids[:-1] for token_ids in token_ids_list]
        y_token_ids = [token_ids[1:] for token_ids in token_ids_list]

        x_token_ids = torch.tensor(np.array(x_token_ids), device= self.device, dtype= torch.long)
        y_token_ids = torch.tensor(np.array(y_token_ids), device= self.device, dtype= torch.long)

        return x_token_ids, y_token_ids
        

    def __len__(self) :
        return len(self.dataset)
