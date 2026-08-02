from __future__ import annotations

import torch
from torch.amp import GradScaler

from train.training_loop import TrainLoop


class _FiniteForwardNanBackward(torch.autograd.Function):
    """模拟前向 loss 有限、反向梯度为 NaN 的数值异常。"""

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return value.sum() * 0.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (value,) = ctx.saved_tensors
        return (torch.full_like(value, float("nan")),)


def _build_minimal_loop(*, gradient_clip: bool) -> TrainLoop:
    loop = TrainLoop.__new__(TrainLoop)
    loop.model = torch.nn.Linear(1, 1, bias=False)
    loop.model.weight.data.fill_(1.0)
    loop.opt = torch.optim.SGD(loop.model.parameters(), lr=0.1)
    loop.scaler = GradScaler("cuda", enabled=False)
    loop.gradient_clip = gradient_clip
    loop.ema_model = None
    loop._anneal_lr = lambda: None
    return loop


def test_finite_gradients_reach_optimizer_step_without_clipping() -> None:
    loop = _build_minimal_loop(gradient_clip=False)

    def forward_backward(_batch) -> None:
        loop.opt.zero_grad(set_to_none=True)
        (loop.model.weight * 2.0).sum().backward()

    loop.forward_backward = forward_backward
    loop.run_step(batch={})

    torch.testing.assert_close(loop.model.weight, torch.tensor([[0.8]]))


def test_nan_gradients_fail_before_optimizer_step() -> None:
    loop = _build_minimal_loop(gradient_clip=False)
    weight_before = loop.model.weight.detach().clone()

    def forward_backward(_batch) -> None:
        loop.opt.zero_grad(set_to_none=True)
        loss = _FiniteForwardNanBackward.apply(loop.model.weight)
        assert torch.isfinite(loss)
        loss.backward()

    loop.forward_backward = forward_backward

    try:
        loop.run_step(batch={})
    except RuntimeError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("NaN 梯度应在 optimizer step 前触发 RuntimeError。")

    torch.testing.assert_close(loop.model.weight, weight_before)
