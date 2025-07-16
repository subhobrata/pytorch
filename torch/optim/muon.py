# mypy: allow-untyped-defs
"""Muon optimizer implementation.

This is a minimal single-device implementation of the Muon optimizer
based on https://github.com/KellerJordan/Muon.
"""

from typing import Iterable, Optional

import torch
from torch import Tensor
from .optimizer import Optimizer, ParamsT

__all__ = ["Muon"]


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """Compute an approximate orthogonalization of ``G`` using
    Newton--Schulz iterations.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def muon_update(
    grad: Tensor,
    momentum: Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(Optimizer):
    """Implements the Muon optimizer for orthogonal weight updates."""

    def __init__(self, params: ParamsT, lr: float = 0.02, weight_decay: float = 0.0, momentum: float = 0.95) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape_as(p), alpha=-group["lr"])

        return loss
