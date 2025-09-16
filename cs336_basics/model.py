import torch
import torch.nn as nn



from cs336_basics.layers import *
from cs336_basics.utils.function import *

class TransformerLM(nn.Module) :
    def __init__(self, vocab_size: int, context_length: int, 
                 num_layers: int, d_model: int, num_heads: int, d_ff: int, 
                 rope_theta: float= 10000.0,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) :
        
        super().__init__()
        self.context_length = context_length

        self.token_embeddings = Embedding(num_embeddings= vocab_size, embedding_dim= d_model, device= device, dtype= dtype)

        layers = [TransformerBlock(d_model= d_model, num_heads= num_heads, d_ff= d_ff, theta= rope_theta, max_seq_len= context_length, device= device, dtype= dtype) for _ in range(num_layers)]
        print("Total layers: ", layers)
        self.layers = nn.Sequential(*layers)

        self.ln_final = RMSNorm(d_model= d_model, device= device, dtype= dtype)
        self.lm_head = Linear(in_features= d_model, out_features= vocab_size)


    def forward(self, indices: torch.Tensor) :
        
        x = self.token_embeddings(indices)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)

        return x
    

    def get_context_length(self) :
        return self.context_length


    
