from __future__ import annotations

import torch

from model.causal_attention import build_frame_causal_mask


def test_frame_causal_mask_blocks_future_frames():
    mask = build_frame_causal_mask(4)
    assert mask.dtype == torch.bool
    assert mask[0, 1:].all()
    assert not mask[3].any()
