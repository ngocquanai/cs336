import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module) :
    def __init__(self) :
        super().__init__()

    def forward(self, logits, target) :
        """
        Notice that CE(x, target_idx) = - log softmax(x)[target_idx]
                                          = log(sum(e**x)) - log e**x[i]) = log(sum(e**x)) - x[i]
        """
        target = target.to(torch.int64)
        # Minus max value for numerical stability
        logits = logits - torch.max(logits, dim= -1, keepdim= True).values
        # breakpoint() 
        
        e_logits = torch.exp(logits)
        log_sum = torch.log(torch.sum(e_logits, dim=-1, keepdim= False))
        logits_idx = logits.gather(dim=-1, index= target.unsqueeze(-1)).squeeze(-1)
        # breakpoint()
        total_loss = log_sum - logits_idx
        return torch.mean(total_loss)

