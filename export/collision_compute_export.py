"""碰撞 Compute Shader 命令链的网络资产和静态常量。"""
import json

import torch

from export.unity_onnx_models import (
    SAMPLER_INPUTS, SamplerCondition, SamplerDenoiseStep, SamplerProjection, example_inputs,
)


def export_collision_chain(sampler, manifest, output_dir):
    inputs = example_inputs()
    condition = SamplerCondition(sampler).eval()
    step = SamplerDenoiseStep(sampler).eval()
    projection = SamplerProjection(sampler).eval()
    cached = condition(*inputs[:8])
    assets = dict(condition='pose_sampler_condition.onnx', step='pose_sampler_step.onnx',
                  projection='pose_sampler_projection.onnx')
    specifications = [
        (condition, inputs[:8], SAMPLER_INPUTS[:8], condition.output_names, 'condition'),
        (step, (inputs[8], sampler.timesteps[:1], *cached),
         ['state', 'timestep', *condition.output_names], ['predicted_residual'], 'step'),
        (projection, (inputs[1], inputs[8], inputs[2]),
         ['predictor_pose_horizon', 'edited_residual', 'current_tracker_raw'], ['deployed_pose'], 'projection'),
    ]
    for module, values, names, outputs, key in specifications:
        torch.onnx.export(module, values, str(output_dir / assets[key]), opset_version=15,
                          dynamo=False, input_names=names, output_names=outputs)
    manifest.pop('sampler', None)
    manifest.update(models=assets, condition_tensors=[dict(name=n, shape=list(v.shape),
        dtype='int32' if v.dtype == torch.int32 else 'float32') for n, v in zip(condition.output_names, cached)],
        sampling_steps=sampler.steps,
        timesteps=sampler.timesteps.flip(0).tolist(),
        ddim=[dict(recip=float(sampler.recip[i]), recipm1=float(sampler.recipm1[i]),
                   sqrt_alpha_prev=float(sampler.alpha_prev[i].sqrt()),
                   sqrt_one_minus_alpha_prev=float((1-sampler.alpha_prev[i]).sqrt()),
                   variance=float(sampler.collision_variance[i])) for i in reversed(range(sampler.steps))],
        pose_mean=sampler.pose_mean.tolist(), pose_scale=sampler.pose_scale.tolist())
    (output_dir / 'collision_compute.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest
