import math
import torch
from einops import rearrange, einsum
import torch.nn as nn
from torch.nn import Parameter
from cs336_basics.utils.function import scale_dot_product_attention, softmax


class Linear(nn.Module) :
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype= None) :
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        std_weight = math.sqrt(2/(in_features + out_features))

        weight = torch.empty((out_features, in_features), device= device, dtype= dtype)
        torch.nn.init.trunc_normal_(weight, mean= 0, std= std_weight, a= -3*std_weight, b= 3*std_weight)
        self.W = Parameter(weight)

    def forward(self, x : torch.Tensor) :
        x = einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
        return x


class Embedding(nn.Module)  :
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None) :
        super().__init__()
        self.num_embeddings = num_embeddings
        self.d_model = embedding_dim
        self.device = device
        self.dtype = dtype

        weight = torch.empty((num_embeddings, self.d_model), dtype= dtype, device= device)
        torch.nn.init.trunc_normal_(weight, mean=0, std= 1, a= -3, b= 3)
        self.W = Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor :
        token_ids_flat = rearrange(token_ids, "batch_size seq_len -> (batch_size seq_len)")
        x = self.W[token_ids_flat, :]
        x = rearrange(x, "(batch_size seq_len) ... -> batch_size seq_len ...", batch_size= token_ids.shape[0], seq_len = token_ids.shape[1])
        return x



class RMSNorm(nn.Module) :
    def __init__(self, d_model: int, eps: float= 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None) :
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        weight = torch.ones(d_model, device= device, dtype= dtype)
        self.weight = Parameter(weight)

    def forward(self, x) :
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_square = torch.square(x)
        scale = 1 / torch.sqrt(torch.sum(x_square, dim= -1) / self.d_model + self.eps)
        
        x = einsum(x, scale, "... d_model, ... -> ... d_model") # x = x /rms_norm(x)
        x = einsum(x, self.weight, "... d_model, d_model -> ... d_model")

        return x.to(in_dtype)


class SwiGLU(nn.Module) :
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None ) :
        super().__init__()
        W1 = torch.empty((d_ff, d_model), device=device, dtype= dtype)
        W3 = torch.empty((d_ff, d_model), device= device, dtype= dtype)
        W2 = torch.empty((d_model, d_ff), device= device, dtype= dtype)

        std_weight = math.sqrt(2/(d_model + d_ff))
        torch.nn.init.trunc_normal_(W1, mean= 0, std= std_weight, a= -3*std_weight, b= 3*std_weight)
        torch.nn.init.trunc_normal_(W2, mean= 0, std= std_weight, a= -3*std_weight, b= 3*std_weight)
        torch.nn.init.trunc_normal_(W3, mean= 0, std= std_weight, a= -3*std_weight, b= 3*std_weight)

        self.W1 = Parameter(W1)
        self.W2 = Parameter(W2)
        self.W3 = Parameter(W3)

    
    def forward(self, x) :
        W1_x = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        W3_x = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")

        x = (W1_x * torch.sigmoid(W1_x)) * W3_x
        x = einsum(x, self.W2, "... d_ff, d_model d_ff -> ... d_model")

        return x


class RoPE(nn.Module) :
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) :
        super().__init__()
        seq_idx = torch.arange(max_seq_len, device = device).float()

        dim_idx = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        angles = einsum(seq_idx, dim_idx, "seq, dim -> seq dim") # max_seq_len, d_k // 2

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos_values", cos, persistent= False)
        self.register_buffer("sin_values", sin, persistent= False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) :

        cos_values = self.cos_values[token_positions]
        sin_values = self.sin_values[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_x_even = x_even * cos_values - x_odd * sin_values 
        rotated_x_odd = x_even * sin_values + x_odd * cos_values

        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = rotated_x_even
        x_rotated[..., 1::2] = rotated_x_odd

        return x_rotated


class MultiHeadSelfAttention_basic_noRoPE(nn.Module) :
    def __init__(self, d_model: int, num_heads: int, device: torch.device|None = None, dtype: torch.dtype|None = None) :
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.query = Linear(in_features= d_model, out_features= int(num_heads*self.d_k), device= device, dtype= dtype)
        self.key = Linear(in_features= d_model, out_features= int(num_heads * self.d_k), device= device, dtype= dtype)
        self.value = Linear(in_features= d_model, out_features= int(num_heads * self.d_v), device= device, dtype= dtype)
        self.output = Linear(in_features= int(num_heads * self.d_v), out_features= d_model, device= device, dtype= dtype)

    def forward(self, x) :
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads= self.num_heads, d_k= self.d_k)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads = self.num_heads, d_k = self.d_k)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads= self.num_heads, d_v = self.d_v)

        seq_len = x.shape[-2]
        mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal= 1).bool()
        attention = scale_dot_product_attention(Q, K, V, mask= mask) # ... num_heads seq_len d_v
        attention = rearrange(attention, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")

        return self.output(attention)

class MultiHeadSelfAttention(nn.Module) :
    def __init__(self, d_model: int, num_heads: int, theta: float=10000.0, max_seq_len: int= 500, device: torch.device|None = None, dtype: torch.dtype|None = None) :
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.QKV = Linear(in_features= d_model, out_features= int(3 * num_heads*self.d_k), device= device, dtype= dtype)
        self.output = Linear(in_features= int(num_heads * self.d_v), out_features= d_model, device= device, dtype= dtype)
        self.rope = RoPE(theta= theta, d_k= self.d_k, max_seq_len= max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) :
        QKV = self.QKV(x)
        Q = QKV[..., 0: int(self.num_heads * self.d_k)]
        K = QKV[..., int(self.num_heads * self.d_k) : int(2* self.num_heads * self.d_k)]
        V = QKV[..., int(2 * self.num_heads * self.d_k):]

        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads= self.num_heads, d_k= self.d_k)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads = self.num_heads, d_k = self.d_k)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads= self.num_heads, d_v = self.d_v)

        # Use rotary position embedding
        Q = self.rope(Q, token_positions= token_positions)
        K = self.rope(K, token_positions= token_positions)

        seq_len = x.shape[-2]
        mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal= 1).bool()
        attention = scale_dot_product_attention(Q, K, V, mask= mask) # ... num_heads seq_len d_v
        attention = rearrange(attention, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")

        return self.output(attention)



class TransformerBlock(nn.Module) :
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float= 10000.0, max_seq_len: int=500, device: torch.device | None = None, dtype: torch.dtype | None = None) :
        super().__init__()

        self.attn_norm = RMSNorm(d_model= d_model, device= device, dtype= dtype)
        self.ffn_norm = RMSNorm(d_model= d_model, device= device, dtype= dtype)
        self.attn = MultiHeadSelfAttention(d_model= d_model, num_heads= num_heads, theta= theta, max_seq_len= max_seq_len, device= device, dtype= dtype)
        self.ffn = SwiGLU(d_model= d_model, d_ff= d_ff, device= device, dtype= dtype)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) :

        if not token_positions :
            seq_len = x.shape[-2]
            token_position = torch.arange(seq_len)
            shape_list = list(x.shape)[:-1] # exclude the last dimension (d_model): ... seq_len
            token_positions = token_position.expand(*shape_list)

        x = x + self.attn(self.attn_norm(x), token_positions)
        x = x + self.ffn(self.ffn_norm(x))

        return x



class TransformerLM(nn.Module) :
    def __init__(
        self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float,
        device: torch.device|None = None, dtype: torch.dtype|None = None
    ) :
        super().__init__()

        self.token_embeddings = Embedding(num_embeddings= vocab_size, embedding_dim= d_model, device= device, dtype= dtype)
        self.layers = nn.Sequential(
            TransformerBlock(
                d_model= d_model, num_heads= num_heads, d_ff= d_ff, theta= rope_theta, max_seq_len= context_length,
                device= device, dtype= dtype
            )
        )
        self.final_norm = RMSNorm(
            d_model= d_model, device= device, dtype= dtype
        )
        self.head = Linear(in_features= d_model, out_features= vocab_size)

    def forward(self, token_ids) :
        x = self.token_embeddings(token_ids)
        x = self.layers(x)
        x = self.final_norm(x)
        x = self.head(x)
        x = softmax(x)

        return x



        

    

        








