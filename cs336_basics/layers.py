import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from einops import rearrange

from cs336_basics.utils.function import scale_dot_product_attention





class Linear(nn.Module) :
    def __init__(self, in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) :
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        sigma = math.sqrt(2/(self.in_features + self.out_features))

        weight = torch.empty(self.out_features, self.in_features, device= device, dtype= dtype)

        torch.nn.init.trunc_normal_(weight, mean=0, std= sigma, a=-3*sigma, b= 3*sigma)

        self.weight = Parameter(weight)

    def init_weight(self, weight) :
        self.weight = Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        return torch.matmul(x, self.weight.T)
    

class Embedding(nn.Module) :
    def __init__(self, num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) :
        super().__init__()

        weight = torch.empty(num_embeddings, embedding_dim, device= device, dtype= dtype)
        torch.nn.init.trunc_normal_(weight, mean=0, std= 1, a= -3, b= 3)

        self.weight = Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor :
        batch = token_ids.shape[0]
        sequence = token_ids.shape[1]

        flat_ids = rearrange(token_ids, "batch sequence -> (batch sequence)")
        flat_embeddings = self.weight[flat_ids, :]
        embeddings = rearrange(flat_embeddings, "(batch sequence) d_model -> batch sequence d_model", batch= batch, sequence= sequence)

        return embeddings
    

class RMSNorm(nn.Module) :
    def __init__(self, d_model: int,
                 eps: float= 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) :
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        weight = torch.ones(self.d_model, device= device, dtype= dtype)
        self.weight = Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm_x = torch.norm(x, p=2, dim= -1) ** 2 # dim_x (..., d_model), dim_norm_x (...)
        rms_x = torch.sqrt(1/self.d_model * norm_x + self.eps) # dim (...)

        rms_norm = x / rms_x.unsqueeze(-1) * self.weight # x (..., d_model), rms_x (...), weight (d_model)
        # breakpoint()
        rms_norm = rms_norm.to(in_dtype)

        return rms_norm
    


class SwiGLU(nn.Module) :
    def __init__(self, d_model: int,
                 d_ff: int | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) :
        super().__init__()

        if not d_ff :
            d_ff = round(8/3 * d_model / 64) * 64

        self.w1 = Linear(in_features= d_model, out_features= d_ff, device= device, dtype= dtype)
        self.w3 = Linear(in_features=d_model, out_features= d_ff, device= device, dtype= dtype)
        self.w2 = Linear(in_features= d_ff, out_features= d_model, device= device, dtype= dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor :
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        x = self.silu(self.w1(x)) * self.w3(x)
        return self.w2(x)
    



# For RoPE class, code from Gemini Pro :)))
class RoPE(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE).

    This module precomputes the sine and cosine values necessary for RoPE
    and applies the rotation to the input tensor during the forward pass.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Initializes the RoPE module.

        Args:
            theta (float): The base value for the geometric progression of frequencies.
                           A value like 10000 is common.
            d_k (int): The dimension of the query and key vectors. Must be even.
            max_seq_len (int): The maximum sequence length that will be processed.
            device (torch.device, optional): The device to store tensors on.
        """
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("The feature dimension d_k must be an even number.")

        # Calculate the inverse frequencies for the rotation.
        # The formula is 1 / (theta^(2k/d_k)) for k = 0, 1, ..., d_k/2 - 1
        # Shape: (d_k / 2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # Create a tensor of positions: [0, 1, ..., max_seq_len-1]
        # Shape: (max_seq_len)
        t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)

        # Calculate the angles for each position and frequency pair using an outer product.
        # Shape: (max_seq_len, d_k / 2)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # Precompute and cache the cosine and sine values.
        # These are registered as non-parameter buffers.
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Tensor of shape (..., seq_len) that specifies
                                            the absolute position of each token in x.

        Returns:
            torch.Tensor: The output tensor with RoPE applied, with the same shape as x.
        """
        # Retrieve the precomputed cos and sin values for the given token positions.
        # After slicing, both cos and sin will have the shape (..., seq_len, d_k / 2).
        cos = self.cos_cached[token_positions] # type: ignore
        sin = self.sin_cached[token_positions] # type: ignore


        # Separate the input tensor x into its even and odd indexed features.
        # This effectively creates pairs of features for rotation.
        # Shape of both x_even and x_odd: (..., seq_len, d_k / 2).
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # breakpoint()


        rotated_x_even = x_even * cos - x_odd * sin
        rotated_x_odd = x_even * sin + x_odd * cos

        # Create an empty tensor with the same shape as the input x
        # to store the result.
        rotated_x = torch.empty_like(x)

        # Interleave the rotated even and odd features back into the output tensor.
        rotated_x[..., 0::2] = rotated_x_even
        rotated_x[..., 1::2] = rotated_x_odd

        return rotated_x
    

# class MultiheadSelfAttention(nn.Module) :
#     def __init__(self, d_model: int, num_heads: int,
#                  device: torch.device | None = None,
#                  dtype: torch.dtype | None = None) :
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads

#         self.d_k = d_model // num_heads


#         self.q_proj = Linear(in_features= d_model, out_features= d_model, device= device, dtype= dtype)
#         self.k_proj = Linear(in_features= d_model, out_features= d_model, device= device, dtype= dtype)
#         self.v_proj = Linear(in_features= d_model, out_features= d_model, device= device, dtype= dtype)
#         self.o_proj = Linear(in_features= d_model, out_features= d_model, device= device, dtype= dtype)

        

#     def forward(self, x: torch.Tensor) : 
#         seq_len = x.shape[-2]
#         mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

#         query, key, value = self.q_proj(x), self.k_proj(x), self.v_proj(x)


#         query = rearrange(query, "... seq (head dk) -> ... head seq dk", head= self.num_heads, dk= self.d_k)
#         key = rearrange(key, "... seq (head dk) -> ... head seq dk", head= self.num_heads, dk= self.d_k)
#         value = rearrange(value, "... seq (head dk) -> ... head seq dk", head= self.num_heads, dk= self.d_k)

#         attention = scale_dot_product_attention(query, key, value, mask= mask)

#         attention = rearrange(attention, "...  head seq dk -> ... seq (head dk)")

#         output = self.o_proj(attention)

#         return output
    


class MultiheadSelfAttention(nn.Module) :
    def __init__(self, d_model: int, num_heads: int,
                 theta: float = 10000.0, max_seq_len: int = 1024,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None, use_rope: bool = True) :
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads

        self.use_rope = use_rope
        self.rope = RoPE(theta= theta, d_k= self.d_k, max_seq_len= max_seq_len, device= device)


        self.q_proj = Linear(in_features= d_model, out_features= d_model, device= device, dtype= dtype)
        self.k_proj = Linear(in_features= d_model, out_features= d_model, device= device, dtype= dtype)
        self.v_proj = Linear(in_features= d_model, out_features= d_model, device= device, dtype= dtype)
        self.output_proj = Linear(in_features= d_model, out_features= d_model, device= device, dtype= dtype)

        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) : 
        seq_len = x.shape[-2] # (... seq dim)
        batch = x.shape[0]
        if token_positions is None :
            token_positions = torch.arange(seq_len)

        mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

        query, key, value = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        query = rearrange(query, "... seq (head dk) -> ... head seq dk", head= self.num_heads, dk= self.d_k)
        key = rearrange(key, "... seq (head dk) -> ... head seq dk", head= self.num_heads, dk= self.d_k)
        value = rearrange(value, "... seq (head dk) -> ... head seq dk", head= self.num_heads, dk= self.d_k)

        # Reshape token position 
        token_positions = token_positions.view(1, 1, -1)
        token_positions= token_positions.expand(batch, self.num_heads, -1) # type: ignore

        if self.use_rope :
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        attention = scale_dot_product_attention(query, key, value, mask= mask)

        attention = rearrange(attention, "...  head seq dk -> ... seq (head dk)")

        output = self.output_proj(attention)

        return output
    

class TransformerBlock(nn.Module) :
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 theta: float = 10000.0, max_seq_len: int = 1024,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) :
        super().__init__()
        self.ln1 = RMSNorm(d_model= d_model, device= device, dtype= dtype)
        self.ln2 = RMSNorm(d_model= d_model, device= device, dtype= dtype)
        self.attn = MultiheadSelfAttention(d_model= d_model, num_heads= num_heads, theta= theta, 
                                           max_seq_len= max_seq_len, device= device, dtype= dtype)
        self.ffn = SwiGLU(d_model= d_model, d_ff= d_ff, device= device, dtype= dtype)

    
    def forward(self, x) :
        x += self.attn(self.ln1(x))
        x += self.ffn(self.ln2(x))

        return x


    

        



    

    


    


