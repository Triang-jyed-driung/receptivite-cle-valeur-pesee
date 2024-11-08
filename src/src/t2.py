
from typing import cast, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch.optim.optimizer import (
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)


class MultiAdamMini(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = True,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = False,
    ):
        foreach = True
        fused = False
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        recip_nelems,
    ):
        has_complex = False
        for p in group["params"]:
            p: Tensor
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("MultiAdamMini does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=group["fused"]),
                        device=p.device,
                    )
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros([],
                    device=p.device, dtype=_get_scalar_dtype()
                )
                # state["recip_nelem"] = torch.tensor(1/p.nelement(), 
                #     device=p.device, dtype=_get_scalar_dtype()
                # )
                state["recip_nelem"] = torch.tensor(1/p.nelement(),
                    device=p.device, dtype=_get_scalar_dtype()
                )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            recip_nelems.append(state["recip_nelem"])

            if group["differentiable"] and state["step"].requires_grad:
                raise RuntimeError(
                    "`requires_grad` is not supported for `step` in differentiable mode"
                )

            # Foreach without capturable does not support a tensor lr
            if (
                group["foreach"]
                and isinstance(group["lr"], Tensor)
                and not group["capturable"]
            ):
                raise RuntimeError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )

            state_steps.append(state["step"])
        return has_complex


    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            recip_nelems = []
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                recip_nelems,
            )
            _multi_tensor_AdamMini(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                recip_nelems,
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
            )

        return loss



# @torch.compile
def _multi_tensor_AdamMini(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    recip_nelems: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None
    if has_complex:
        _view_as_real(
            params, grads, exp_avgs, exp_avg_sqs
        )

    if maximize:
        grads = torch._foreach_neg(grads)  # type: ignore[assignment]

    # Update steps
    # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
    # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
    # wrapped it once now. The alpha is required to assure we go to the right overload.
    # if not torch._utils.is_compiling() and state_steps[0].is_cpu:
    #     torch._foreach_add_(
    #         state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
    #     )
    # else:
    torch._foreach_add_(state_steps, 1)

    # Perform stepweight decay
    if weight_decay != 0:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
    
    # Decay the first and second moment running average coefficient
    torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)

    grad_2 = torch._foreach_norm(grads, ord=2)
    del grads
    # sqrt(x1^2+x2^2+...+xn^2)

    torch._foreach_mul_(grad_2, grad_2)
    # x1^2+x2^2+...+xn^2

    torch._foreach_mul_(grad_2, recip_nelems)
    # (x1^2+x2^2+...+xn^2) * (1/n)

    torch._foreach_lerp_(exp_avg_sqs, grad_2, 1 - beta2)


    # bias_correction1 = torch._foreach_pow(beta1, state_steps)
    # bias_correction2 = torch._foreach_pow(beta2, state_steps)
    # # foreach_sub doesn't allow a scalar as the first arg
    # torch._foreach_sub_(bias_correction1, 1)
    # torch._foreach_sub_(bias_correction2, 1)
    # # we do not negate bias_correction1 as it'll need to be negated later anyway
    # torch._foreach_neg_(bias_correction2)

    # # foreach_div doesn't allow a scalar as the first arg
    # torch._foreach_div_(bias_correction1, lr)
    # torch._foreach_reciprocal_(bias_correction1)

    # torch._foreach_sqrt_(bias_correction2)

    # # Re-assign for clarity as we maintain minimal intermediates: we'll have
    # # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
    # # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
    # step_size = bias_correction1
    # bias_correction2_sqrt = bias_correction2

    # exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)

    
    # torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
    # torch._foreach_add_(exp_avg_sq_sqrt, eps)
    # torch._foreach_div_(exp_avg_sq_sqrt, step_size)

    # # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
    # torch._foreach_addcdiv_(params, exp_avgs, exp_avg_sq_sqrt)
    
    bias_correction1 = [
        1 - beta1 ** _get_value(step) for step in state_steps
    ]
    bias_correction2 = [
        1 - beta2 ** _get_value(step) for step in state_steps
    ]

    step_size = [(lr / bc) * -1 for bc in bias_correction1]
    bias_correction2_sqrt = [
        bc**0.5 for bc in bias_correction2  # type: ignore[arg-type]
    ]

    exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
        
    torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
    torch._foreach_add_(exp_avg_sq_sqrt, eps)
    torch._foreach_addcdiv_(
        params,
        exp_avgs,
        exp_avg_sq_sqrt,
        step_size,
    )



    # grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
    #     [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, recip_nelems]  # type: ignore[list-item]
    # )
    # for (
    #     device_params_,
    #     device_grads_,
    #     device_exp_avgs_,
    #     device_exp_avg_sqs_,
    #     device_max_exp_avg_sqs_,
    #     device_state_steps_,
    #     device_recip_nelems_,
    # ), _ in grouped_tensors.values():
    #     device_params = cast(List[Tensor], device_params_)
    #     device_grads = cast(List[Tensor], device_grads_)
    #     device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
    #     device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
    #     device_state_steps = cast(List[Tensor], device_state_steps_)
    #     device_recip_nelems = cast(List[Tensor], device_recip_nelems_)

    #     if has_complex:
    #         if amsgrad:
    #             device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)
    #             _view_as_real(
    #                 device_params,
    #                 device_grads,
    #                 device_exp_avgs,
    #                 device_exp_avg_sqs,
    #                 device_max_exp_avg_sqs,
    #             )
    #         else:
    #             _view_as_real(
    #                 device_params, device_grads, device_exp_avgs, device_exp_avg_sqs
    #             )

    #     if maximize:
    #         device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

    #     # Update steps
    #     # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
    #     # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
    #     # wrapped it once now. The alpha is required to assure we go to the right overload.
    #     if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:
    #         torch._foreach_add_(
    #             device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
    #         )
    #     else:
    #         torch._foreach_add_(device_state_steps, 1)

    #     # Perform stepweight decay
    #     if weight_decay != 0:
    #         torch._foreach_mul_(device_params, 1 - lr * weight_decay)
        
    #     # Decay the first and second moment running average coefficient
    #     torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

    #     device_grad_2 = torch._foreach_norm(device_grads_, ord=2)
    #     del device_grads
    #     # sqrt(x1^2+x2^2+...+xn^2)

    #     torch._foreach_mul_(device_grad_2, device_grad_2)
    #     # x1^2+x2^2+...+xn^2

    #     torch._foreach_mul_(device_grad_2, device_recip_nelems)
    #     # (x1^2+x2^2+...+xn^2) * (1/n)

    #     torch._foreach_lerp_(device_exp_avg_sqs, device_grad_2, 1 - beta2)
        
    #     bias_correction1: Union[Tuple[Tensor, ...], List[Tensor]]
    #     bias_correction2: Union[Tuple[Tensor, ...], List[Tensor]]
    #     bias_correction2_sqrt: Union[Tuple[Tensor, ...], List[Tensor]]

    #     if False:
    #         bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
    #         bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
    #         # foreach_sub doesn't allow a scalar as the first arg
    #         torch._foreach_sub_(bias_correction1, 1)
    #         torch._foreach_sub_(bias_correction2, 1)
    #         # we do not negate bias_correction1 as it'll need to be negated later anyway
    #         torch._foreach_neg_(bias_correction2)

    #         # foreach_div doesn't allow a scalar as the first arg
    #         torch._foreach_div_(bias_correction1, lr)
    #         torch._foreach_reciprocal_(bias_correction1)

    #         torch._foreach_sqrt_(bias_correction2)

    #         # Re-assign for clarity as we maintain minimal intermediates: we'll have
    #         # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
    #         # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
    #         step_size = bias_correction1
    #         bias_correction2_sqrt = bias_correction2

    #         if amsgrad:
    #             device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)

    #             # Maintains the maximum of all 2nd moment running avg. till now
    #             torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

    #             # Use the max. for normalizing running avg. of gradient
    #             exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
    #         else:
    #             exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            
    #         torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
    #         torch._foreach_add_(exp_avg_sq_sqrt, eps)
    #         torch._foreach_div_(exp_avg_sq_sqrt, step_size)

    #         # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
    #         torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
    #     else:
    #         bias_correction1 = [
    #             1 - beta1 ** _get_value(step) for step in device_state_steps
    #         ]
    #         bias_correction2 = [
    #             1 - beta2 ** _get_value(step) for step in device_state_steps
    #         ]

    #         step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

    #         bias_correction2_sqrt = [
    #             bc**0.5 for bc in bias_correction2  # type: ignore[arg-type]
    #         ]

    #         if amsgrad:
    #             device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)

    #             # Maintains the maximum of all 2nd moment running avg. till now
    #             torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

    #             # Use the max. for normalizing running avg. of gradient
    #             exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
    #         else:
    #             exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
                
    #         torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
    #         torch._foreach_add_(exp_avg_sq_sqrt, eps)
    #         torch._foreach_addcdiv_(
    #             device_params,
    #             device_exp_avgs,
    #             exp_avg_sq_sqrt,
    #             step_size,  # type: ignore[arg-type]
    #         )


# # Identical function.
# @torch.compile
# def _multi_tensor_AdamMini2(
#     params: List[Tensor],
#     grads: List[Tensor],
#     exp_avgs: List[Tensor],
#     exp_avg_sqs: List[Tensor],
#     state_steps: List[Tensor],
#     recip_nelems: List[Tensor],
#     grad_scale: Optional[Tensor],
#     found_inf: Optional[Tensor],
#     *,
#     beta1: float,
#     beta2: float,
#     lr: Union[Tensor, float],
#     weight_decay: float,
#     eps: float,
#     maximize: bool,
#     capturable: bool,
#     differentiable: bool,
#     has_complex: bool,
# ):
#     if len(params) == 0:
#         return

#     if isinstance(lr, Tensor) and not capturable:
#         raise RuntimeError(
#             "lr as a Tensor is not supported for capturable=False and foreach=True"
#         )

#     # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
#     if not torch._utils.is_compiling() and capturable:
#         capturable_supported_devices = _get_capturable_supported_devices(
#             supports_xla=False
#         )
#         assert all(
#             p.device.type == step.device.type
#             and p.device.type in capturable_supported_devices
#             for p, step in zip(params, state_steps)
#         ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

#     assert not differentiable, "_foreach ops don't support autograd"

#     assert grad_scale is None and found_inf is None
#     if has_complex:
#         _view_as_real(
#             params, grads, exp_avgs, exp_avg_sqs
#         )

#     if maximize:
#         grads = torch._foreach_neg(grads)  # type: ignore[assignment]

#     # Update steps
#     # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
#     # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
#     # wrapped it once now. The alpha is required to assure we go to the right overload.
#     # if not torch._utils.is_compiling() and state_steps[0].is_cpu:
#     #     torch._foreach_add_(
#     #         state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
#     #     )
#     # else:
#     torch._foreach_add_(state_steps, 1)

#     # Perform stepweight decay
#     if weight_decay != 0:
#         torch._foreach_mul_(params, 1 - lr * weight_decay)
    
#     # Decay the first and second moment running average coefficient
#     torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)

#     grad_2 = torch._foreach_norm(grads, ord=2)
#     del grads
#     # sqrt(x1^2+x2^2+...+xn^2)

#     torch._foreach_mul_(grad_2, grad_2)
#     # x1^2+x2^2+...+xn^2

#     torch._foreach_mul_(grad_2, recip_nelems)
#     # (x1^2+x2^2+...+xn^2) * (1/n)

#     torch._foreach_lerp_(exp_avg_sqs, grad_2, 1 - beta2)


#     bias_correction1 = torch._foreach_pow(beta1, state_steps)
#     bias_correction2 = torch._foreach_pow(beta2, state_steps)
#     # foreach_sub doesn't allow a scalar as the first arg
#     torch._foreach_sub_(bias_correction1, 1)
#     torch._foreach_sub_(bias_correction2, 1)
#     # we do not negate bias_correction1 as it'll need to be negated later anyway
#     torch._foreach_neg_(bias_correction2)

#     # foreach_div doesn't allow a scalar as the first arg
#     torch._foreach_div_(bias_correction1, lr)
#     torch._foreach_reciprocal_(bias_correction1)

#     torch._foreach_sqrt_(bias_correction2)

#     # Re-assign for clarity as we maintain minimal intermediates: we'll have
#     # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
#     # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
#     step_size = bias_correction1
#     bias_correction2_sqrt = bias_correction2

#     exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)

    
#     torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
#     torch._foreach_add_(exp_avg_sq_sqrt, eps)
#     torch._foreach_div_(exp_avg_sq_sqrt, step_size)

#     # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
#     torch._foreach_addcdiv_(params, exp_avgs, exp_avg_sq_sqrt)