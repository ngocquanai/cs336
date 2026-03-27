import torch

x = torch.ones(2,3,4,5)
token_position = torch.arange(4)
shape_list = list(x.shape)[:-1] # exclude the last dimension (d_model): ... seq_len
token_positions = token_position.expand(*shape_list)
breakpoint()
