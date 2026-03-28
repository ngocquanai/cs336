import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

class SGD(torch.optim.Optimizer) :
    def __init__(self, params, lr= 1e-3) :
        if lr < 0 :
            raise ValueError(f"Invalid Learning rate values: {lr}")
        
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) :
        loss = None if closure is None else closure()

        for group in self.param_groups :
            lr = group["lr"]
            for p in group["params"] :
                if p.grad is None :
                    continue
            
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr/math.sqrt(t+1) * grad
                state["t"] = t + 1

        return loss

class AdamW(torch.optim.Optimizer) :
    def __init__(self, params, lr: float= 1e-3, betas: tuple[float] = (0.9, 0.99), weight_decay: float= 0.1, eps: float= 1e-8) :
        if lr < 0 :
            raise ValueError(f"Invalid learning rate value: {lr}")
        
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) :
        loss = None if closure is None else closure()

        for group in self.param_groups :
            lr = group["lr"]
            beta_1 = group["betas"][0]
            beta_2 = group["betas"][1]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"] :
                if p.grad is None :
                    continue
                state = self.state[p]
                first_moment = state.get("first_moment", 0)
                second_moment = state.get("second_moment", 0)
                t = state.get("t", 1)

                grad = p.grad.data
                first_moment = beta_1 * first_moment + (1 - beta_1) * grad
                second_moment = beta_2 * second_moment + (1 - beta_2) * grad**2

                adjusted_lr = lr * ( math.sqrt(1 - beta_2**t) / (1 - beta_1**t ))

                p.data -= adjusted_lr * (first_moment / (torch.sqrt(second_moment) + eps)) # grad update
                p.data -= lr * wd * p.data # weight decay update

                # Update state
                state["t"] = t + 1
                state["first_moment"] = first_moment
                state["second_moment"] = second_moment




