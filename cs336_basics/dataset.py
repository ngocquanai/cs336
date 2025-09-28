import random
import torch
import numpy as np
import os

def get_batch(x, batch_size, context_length, device) :

    total_tokens = x.shape[0]
    inputs = []
    labels = []
    for _ in range(batch_size) :
        # Choose a random valid sample
        start = random.randint(0, total_tokens - context_length - 1)
        inputs.append(x[start:start+context_length])
        labels.append(x[start+1:start+1+context_length])
    
    inputs = np.stack(inputs, axis= 0)
    labels = np.stack(labels, axis= 0)

    inputs = torch.from_numpy(inputs).to(device= device, dtype= torch.int64)
    labels = torch.from_numpy(labels).to(device= device, dtype= torch.int64)
    return inputs, labels


class DatasetLM :
    def __init__(self, train_path, valid_path, batch_size, context_len,
                 device: torch.device | None = None) -> None:
        self.train = np.memmap(filename= train_path, dtype= np.int32, mode="r")
        self.val = np.memmap(filename= valid_path, dtype= np.int32, mode= "r")
        self.batch_size = batch_size
        self.context_len = context_len
        self.device = device

    def sample(self, split= "train", batch_size: int | None = None) :
        if batch_size is None :
            batch_size = self.batch_size

        if split == "train" :
            batch = get_batch(self.train, batch_size= batch_size, 
                             context_length= self.context_len, device= self.device)
        else :
            batch = get_batch(self.val, batch_size= batch_size,
                             context_length= self.context_len, device= self.device)

        return batch
        
    def get_len(self, split= "train") :
        if split == "train" :
            return self.train.size
        else :
            return self.val.size
        



    





    