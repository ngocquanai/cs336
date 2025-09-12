import torch
import math


from collections.abc import Callable, Iterable
from typing import Optional


class AdamW(torch.optim.Optimizer) :
    def __init__(self, params, lr: float = 1e-4, 
                 betas: tuple = (0.9, 0.999), weight_decay: float = 1e-5, eps: float = 1e-8) :
        if lr < 0 :
            raise ValueError(f"Invalid learning rate: {lr}")
        elif weight_decay < 0 :
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        
        defaults = {"lr": lr, "beta": betas, "wd": weight_decay}

        super().__init__(params= params, defaults= defaults)
        self.eps = eps
    

    def step(self, closure: Optional[Callable]= None) : # type: ignore
        if closure is None :
            loss = None
        else :
            loss = closure()
        
        for group in self.param_groups :
            lr = group["lr"]
            beta1, beta2 = group["beta"]
            wd = group["wd"]


            for p in group["params"] :
                if p.grad is None :
                    continue
                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                time = state.get("time", 1)


                grad = p.grad.data

                m = beta1 * m + (1-beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                
                adjusted_lr = lr * math.sqrt(1 - (beta2)**time) / (1 - (beta1)**time)

                p.data -= adjusted_lr * (m / (torch.sqrt(v) + self.eps)) # grad update
                p.data -= lr * wd * p # weight decay update

                # Save new state
                state["m"] = m
                state["v"] = v
                state["time"] = time + 1
                
        
        return loss
    




    