from __future__ import annotations

import numpy as np
import pytest
import torch

from export import export_sentis_denoiser
from export.unity_onnx_models import SamplerLayerNorm
from scripts import export_smpl_source_rest


def test_export_smpl_source_rest_imports_and_builds_parser():
    parser = export_smpl_source_rest.build_arg_parser()
    args = parser.parse_args(
        ["--source_npz", "source.npz", "--amass_path", "subject_motion.npz"]
    )
    assert args.source_npz == "source.npz"


def test_source_rest_offsets_keep_positive_grounded_pelvis():
    joints = np.zeros((24, 3), dtype=np.float64)
    for joint_index in range(1, 24):
        joints[joint_index] = joints[joint_index - 1] + np.asarray([0.0, 0.1, 0.0])
    joints[:, 1] += 1.0
    vertices = np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)

    rest_offsets, source_fk_offsets = export_smpl_source_rest.build_rest_local_offsets(
        joints,
        vertices,
    )

    assert rest_offsets.shape == (24, 3)
    assert source_fk_offsets.shape == (24, 3)
    assert rest_offsets[0, 1] == 1.0


def test_unity_export_parser_accepts_two_models():
    args = export_sentis_denoiser.build_arg_parser().parse_args([
        '--model_path', 'dit.pt', '--predictor_model_path', 'predictor.pt',
        '--normalizer_dir', 'normalizer', '--body_fbx_rest_json', 'rest.json',
    ])
    assert str(args.predictor_model_path) == 'predictor.pt'
    assert args.ts_respace == '10'


@pytest.mark.parametrize('shape', [(1, 24, 192), (24, 1, 192)])
def test_sampler_layer_norm_preserves_adaln_math(shape):
    generator = torch.Generator().manual_seed(10)
    value = torch.randn(shape, generator=generator)
    scale = torch.randn(shape[0], 1, shape[-1], generator=generator)
    shift = torch.randn(shape[0], 1, shape[-1], generator=generator)
    reference = torch.nn.LayerNorm(shape[-1], elementwise_affine=False, eps=1e-5)
    actual = SamplerLayerNorm(reference.eps)(value) * (1 + scale) + shift
    expected = reference(value) * (1 + scale) + shift
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
