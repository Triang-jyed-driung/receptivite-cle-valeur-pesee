import torch
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import _get_scalar_dtype
import wandb

class MyMini(Optimizer):
    # An "over" simplified version of Adam-Mini, but should work.
    def __init__(
        self,
        model=None,
        lr=1e-3,
        betas=(0.9,0.999),
        eps=1e-18,
        weight_decay=0.001,
        # specify which names to decay
        weight_decay_condition=lambda name, param: True,
        # logging
        logging_fn=lambda x: None,
        logging_steps = -1
    ):
        self.model = model
        self.logging = logging_fn
        self.logging_steps = logging_steps
        optim_groups = [
            {
                "name": name, 
                "params": p, 
                "weight_decay": weight_decay_condition(name, p) * weight_decay
            }
            for name, p in self.model.named_parameters()
            if p.requires_grad
        ]
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(MyMini, self).__init__(optim_groups, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                name = group["name"]
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                eps = group["eps"]

                # for p in group["params"]:
                for index in range(len(group["params"])):
                    p: torch.Tensor = group["params"][index]
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("Adam-Mini does not support sparse gradients")
                    state = self.state[p]
                    # State initialization
                    if (len(state)==0):                   
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of scalar averaged squared gradient values
                        state["exp_avg_sq"] = torch.zeros([1], device=p.device, dtype=_get_scalar_dtype())

                    # update step
                    state["step"] += 1
                    # apply weight decay
                    p.mul_(1 - lr*group["weight_decay"])
                    # Decay the first and second moment running average coefficient
                    state["exp_avg"].lerp_(p.grad, 1-beta1)
                    state["exp_avg_sq"].lerp_(p.grad.square_().mean(), 1-beta2)

                    step = state["step"]
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    step_size = lr / bias_correction1
                    bias_correction2_sqrt = bias_correction2**0.5
                    denom = (state["exp_avg_sq"].sqrt() / bias_correction2_sqrt).add_(eps)
                    # update parameters
                    p.addcdiv_(state["exp_avg"], denom, value=-step_size)

                    if state["step"] % 40 == 1: # self.logging_steps > 0 and state["step"] % self.logging_steps == 1:
                        wandb.log({f"u{name}_{p.nelement()}".replace('.', ''): float(step_size / denom)}, commit=False)
                        wandb.log({f"m{name}_{p.nelement()}".replace('.', ''): float(state["exp_avg_sq"])}, commit=False)
                        

        return loss
    

class MyMiniForEach(Optimizer):
    
    # A for-each version of Adam-Mini, but should work.
    def __init__(
        self,
        model=None,
        lr=1e-3,
        betas=(0.9,0.999),
        eps=1e-18,
        weight_decay=0.001,
        # specify which names to decay
        weight_decay_condition=lambda name, param: True,
        # logging
        logging_fn=lambda x: None,
        logging_steps = -1
    ):
        raise NotImplementedError
        self.model = model
        self.logging = logging_fn
        self.logging_steps = logging_steps
        names=[]
        ps=[]
        navgs=[]
        nodecay_names=[]
        nodecay_ps=[]
        nodecay_navgs=[]
        for name, p in self.named_parameters():
            if p.requires_grad:
                if weight_decay_condition(name, p):
                    names.append(name)
                    ps.append(p)
                    navgs.append(torch.tensor([1/p.nelement()], dtype=p.dtype, device=p.device))
                else:
                    nodecay_names.append(name)
                    nodecay_ps.append(p)
                    nodecay_navgs.append(torch.tensor([1/p.nelement()], dtype=p.dtype, device=p.device))
        optim_groups = [
            {"name": names, "params": ps, "navg": navgs, "weight_decay": weight_decay},
            {"name": nodecay_names, "params": nodecay_ps, "navg": nodecay_navgs, "weight_decay": 0.0},
        ]
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(MyMiniForEach, self).__init__(optim_groups, defaults)

    def step(self, closure=None):
        raise NotImplementedError
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                name = group["name"]
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                eps = group["eps"]

                # for p in group["params"]:
                for index in range(len(group["params"])):
                    p: torch.Tensor = group["params"][index]
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("Adam-Mini does not support sparse gradients")
                    state = self.state[p]
                    # State initialization
                    if (len(state)==0):                   
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of scalar averaged squared gradient values
                        state["exp_avg_sq"] = torch.zeros([1], device=p.device, dtype=_get_scalar_dtype())

                    # update step
                    state["step"] += 1
                    # apply weight decay
                    p.mul_(1 - lr*group["weight_decay"])
                    # Decay the first and second moment running average coefficient
                    state["exp_avg"].lerp_(p.grad, 1-beta1)
                    state["exp_avg_sq"].lerp_(p.grad.square_().mean(), 1-beta2)

                    step = state["step"]
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    step_size = lr / bias_correction1
                    bias_correction2_sqrt = bias_correction2**0.5
                    denom = (state["exp_avg_sq"].sqrt() / bias_correction2_sqrt).add_(eps)
                    # update parameters
                    p.addcdiv_(state["exp_avg"], denom, value=-step_size)

                    if state["step"] % 40 == 1: # self.logging_steps > 0 and state["step"] % self.logging_steps == 1:
                        wandb.log({f"u{name}_{p.nelement()}".replace('.', ''): float(step_size / denom)}, commit=False)
                        wandb.log({f"m{name}_{p.nelement()}".replace('.', ''): float(state["exp_avg_sq"])}, commit=False)
                        

        return loss