import math
import torch
from einops import rearrange,einsum

def softmax(x: torch.Tensor, dim: int= -1) :
    x = x - torch.max(x, dim= dim, keepdim= True).values # trick for numerical stability
    exp_x = torch.exp(x)

    return exp_x / torch.sum(exp_x, dim= dim, keepdim= True)


def scale_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) :
    d_k = K.shape[-1]
    d_v = V.shape[-1]

    pre_softmax = einsum(Q, K, "... n d_k, ... m d_k -> ... n m")/ math.sqrt(d_k) # (..., n, m); mask (n, m)

    if mask is not None :
        pre_softmax = pre_softmax.masked_fill(~mask, float('-inf')) # Inverse of mask, not mask
        # breakpoint()
    
    after_softmax = softmax(pre_softmax, dim= -1)

    output = einsum(after_softmax, V, "... n m, ... m d_v -> ... n d_v")
    return output


def gradient_clipping(parameters, max_norm, eps=1e-6) :
    total_norm = sum([torch.sum(param.grad ** 2) for param in parameters if param.grad is not None])
    total_norm = total_norm ** 0.5

    if total_norm > max_norm :
        for param in parameters :
            if param.grad is None :
                continue
            param.grad.data = param.grad.data /(total_norm + eps) * max_norm


        








