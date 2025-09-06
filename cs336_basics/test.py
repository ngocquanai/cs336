# import os
# from typing import IO, Any, BinaryIO
# from collections.abc import Iterable
# from jaxtyping import Float, Int

# import numpy.typing as npt
# import torch
# from torch import Tensor
# from cs336_basics.function import scale_dot_product_attention
# from cs336_basics.layers import *


# def run_multihead_self_attention(
#     d_model: int,
#     num_heads: int,
#     q_proj_weight: Float[Tensor, " d_k d_in"],
#     k_proj_weight: Float[Tensor, " d_k d_in"],
#     v_proj_weight: Float[Tensor, " d_v d_in"],
#     o_proj_weight: Float[Tensor, " d_model d_v"],
#     in_features: Float[Tensor, " ... sequence_length d_in"],
# ) -> Float[Tensor, " ... sequence_length d_out"]:
#     """
#     Given the key, query, and value projection weights of a naive unbatched
#     implementation of multi-head attention, return the output of an optimized batched
#     implementation. This implementation should handle the key, query, and value projections
#     for all heads in a single matrix multiply.
#     This function should not use RoPE.
#     See section 3.2.2 of Vaswani et al., 2017.

#     Args:
#         d_model (int): Dimensionality of the feedforward input and output.
#         num_heads (int): Number of heads to use in multi-headed attention.
#         max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
#         q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
#         k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
#         v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
#         o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
#         in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

#     Returns:
#         Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
#         implementation with the given QKV projection weights and input features.
#     """
#     multihead_attn = MultiheadSelfAttention(d_model= d_model, num_heads= num_heads)
#     weights = {"q_proj.W": q_proj_weight, "k_proj.W": k_proj_weight, "v_proj.W": v_proj_weight, "o_proj.W": o_proj_weight}
#     multihead_attn.load_state_dict(weights)

#     return multihead_attn(in_features)

# b = 10

# d_in = 64
# d_model = 64
# num_heads = 4




# q_proj = torch.rand(d_model, d_model)
# k_proj = torch.rand(d_model, d_model)
# v_proj = torch.rand(d_model, d_model)


# o_proj = torch.rand(d_model, d_model)

# x = torch.rand(b, 120, d_in)

# ans = run_multihead_self_attention(d_model, num_heads, q_proj, k_proj, v_proj, o_proj, x)



import torch

# Original tensor of shape (, d)
d = 5
original_tensor = torch.randn(d) 

# Desired batch size
b = 3

# Unsqueeze to add a batch dimension of size 1, then expand
broadcasted_tensor = original_tensor.unsqueeze(0).expand(b, -1)

print(f"Original tensor shape: {original_tensor.shape}")
print(f"Broadcasted tensor shape: {broadcasted_tensor.shape}")
