"""只检查本次部署时间 Attention 的实际形状和 ONNX 数值。"""
from copy import deepcopy

import numpy as np
import pytest
import torch

from export.unity_onnx_models import SamplerTemporalAttention


class ReusedAttention(torch.nn.Module):
    """小图显式准备本次 K/V，检查时保留 K/V 输出便于发现缓存错误。"""
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, query, context):
        kv = self.attention.prepare_kv(context)
        return self.attention(query, kv), *kv


@pytest.mark.parametrize('near_constant', [False, True])
def test_temporal_attention_deployment_matches_mha(tmp_path, near_constant):
    ort = pytest.importorskip('onnxruntime')
    torch.manual_seed(10)
    original = torch.nn.MultiheadAttention(192, 6, batch_first=True).eval()
    candidate = ReusedAttention(SamplerTemporalAttention(deepcopy(original))).eval()
    query = torch.randn(24, 1, 192)
    context = torch.randn(24, 20, 192)
    if near_constant:
        query = 1 + query * 1e-6
        context = 1 + context * 1e-6
    with torch.inference_mode():
        expected = original(query, context, context, need_weights=False)[0]
        actual = candidate(query, context)[0]
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
        path = tmp_path / 'temporal.onnx'
        torch.onnx.export(candidate, (query, context), str(path),
                          opset_version=15, dynamo=False)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    # 小图只接收 query/context；K/V 在图内准备，不作为跨调用状态传入。
    feed = dict(zip((item.name for item in session.get_inputs()),
                    (query.numpy(), context.numpy())))
    np.testing.assert_allclose(session.run(None, feed)[0], expected.numpy(), atol=1e-4, rtol=1e-3)


def test_temporal_kv_recomputed_between_calls_and_reused_for_changing_query():
    torch.manual_seed(10)
    original = torch.nn.MultiheadAttention(192, 6, batch_first=True).eval()
    candidate = SamplerTemporalAttention(deepcopy(original)).eval()
    query = torch.randn(24, 1, 192)
    contexts = [torch.randn(24, 20, 192), torch.randn(24, 20, 192)]
    with torch.inference_mode():
        first = candidate.prepare_kv(contexts[0])
        for context in (contexts[0], contexts[1], contexts[0]):
            kv = candidate.prepare_kv(context)
            context_sequence = context.transpose(0, 1)
            _, key, value = torch.nn.functional._in_projection_packed(
                query.transpose(0, 1), context_sequence, context_sequence,
                original.in_proj_weight, original.in_proj_bias)
            # 用原投影与原 head 排布产生参考；归一化和缩放轴不变。
            key = key.view(20, 144, 32).transpose(0, 1).view(24, 6, 20, 32)
            value = value.view(20, 144, 32).transpose(0, 1).view(24, 6, 20, 32)
            scale = key.new_tensor(32.).sqrt().reciprocal().sqrt()
            torch.testing.assert_close(kv[0], key * scale, atol=1e-4, rtol=1e-3)
            torch.testing.assert_close(kv[1], value.transpose(-2, -1), atol=1e-4, rtol=1e-3)
            for current_query in (query, query + .1):
                expected = original(current_query, context, context, need_weights=False)[0]
                torch.testing.assert_close(candidate(current_query, kv), expected, atol=1e-4, rtol=1e-3)
        for actual, expected in zip(kv, first):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
