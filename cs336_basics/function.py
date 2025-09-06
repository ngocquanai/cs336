import torch
import math


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






    
