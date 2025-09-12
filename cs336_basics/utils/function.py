import torch
import math
import torch.nn.functional as F
import numpy as np
import random

def softmax(x: torch.Tensor, dim: int = -1) :

    x = x - torch.max(x, dim=dim)[0].unsqueeze(dim)
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim).unsqueeze(dim)
    return x


def scale_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                mask: torch.Tensor | None = None) :
    d_k = key.shape[-1]
    d_v = value.shape[-1]
    qk = torch.matmul(query, key.transpose(-2, -1))
    qk = qk / torch.tensor(math.sqrt(d_k))
    if mask is not None :
        qk = qk.masked_fill(~mask, torch.tensor(float("-inf")))

    qk_softmax = softmax(qk, dim= -1)
    attention = torch.matmul(qk_softmax, value)

    return attention


def cross_entropy(x, target) :
    loss = F.cross_entropy(x, target)

    return loss



def learning_rate_schedule(t, a_min, a_max, T_w, T_c) :
    if t < T_w : # warm up
        a_t = t / T_w * a_max
        return a_t

    elif t > T_c : # constant
        a_t = a_min
        return a_t
    
    # ELSE
    a_t = a_min + 1/2*(a_max - a_min) * (1 + math.cos(math.pi * ((t - T_w)/(T_c - T_w))))

    return a_t

def gradient_clipping(params, max_norm, eps= 1e-6) :

    # Total norm square
    total_norm_square = sum([torch.sum(p.grad**2) for p in params if p.grad is not None])
    total_norm = torch.sqrt(total_norm_square) # type: ignore

    if total_norm > max_norm :
        scale = max_norm / (total_norm + eps)
        for p in params :
            if p.grad is not None :
                p.grad.detach().mul_(scale)



def data_loading(x, batch_size, context_length, device) :

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


    


    

    







    
