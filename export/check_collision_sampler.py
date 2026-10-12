"""用真实录制条件验证完整碰撞 sampler，并生成 Unity GPU 对照输入。"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from diffusion.realtime_pose_inpainting import build_current_realtime_pose_conditions
from diffusion.realtime_pose_projection import project_realtime_pose_xstart
from export.export_sentis_denoiser import (
    COLLISION_KEYS, IK_KEYS, build_arg_parser, build_collision_postprocess, load_models,
)
from export.residual_collision_postprocess import COLLISION_INPUTS
from export.unity_onnx_models import PoseSampler, SAMPLER_INPUTS
from export.check_collision_chain import CollisionChainReference
from sample import infer_unity_recording as bridge
from sample.realtime_pose_runtime import RealtimePoseRuntime
from utils.parser_util import parse_and_load_from_model


def main(argv=None):
    parser = build_arg_parser()
    parser.add_argument('--input', type=Path, default=Path('sample/assets/unity_recordings/recording_20260829_164358.json'))
    args = parse_and_load_from_model(parser, argv, ignore_keys={
        'normalizer_dir', 'output_dir', 'body_fbx_rest_json', 'input', 'sampler_only', 'sampler_filename'} | COLLISION_KEYS)
    if not args.collision_postprocess or args.sampler_only:
        raise ValueError('对照需要 --collision_postprocess，且不能传 --sampler_only。')
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    predictor, dit, diffusion, normalizer = load_models(args)
    post, _ = build_collision_postprocess(args, normalizer)
    sampler = PoseSampler(dit, diffusion, normalizer, post).eval()
    normal = PoseSampler(dit, diffusion, normalizer).eval()
    session = CollisionChainReference(args.output_dir, post)
    recording = bridge.resample_tracker_recording(bridge.apply_tracker_availability_overrides(
        bridge.load_unity_tracker_recording(args.input), ignore_hip=True, ignore_feet=False))
    rest = bridge.load_body_fbx_rest(args.body_fbx_rest_json)
    runtime = RealtimePoseRuntime(predictor, dit, diffusion, torch.device('cpu'), rest.rest_local_positions,
        bridge.rotation_6d_forward_up_np(rest.rest_local_rotations), normalizer, **{k: getattr(args, k) for k in IK_KEYS})
    start = bridge.find_first_core_window(recording.available)
    runtime.initialize_history(bridge.build_bootstrap_pose_history(recording, rest, start),
        recording.positions[start:start+11], recording.rotations_6d[start:start+11], recording.floor_y)
    generator = torch.Generator().manual_seed(10)
    rows = []
    with torch.inference_mode():
        for tick in range(3):
            current = start + 11 + tick
            prepared = runtime._prepare_step(recording.positions[current], recording.rotations_6d[current],
                                              recording.available[current], recording.floor_y)
            motion = torch.from_numpy(normalizer.normalize_pose(prepared.motion_context[None]).astype(np.float32))
            sparse = normalizer.normalize_predictor_sparse(prepared.core_tracker_context[None])
            sparse = torch.from_numpy(np.where(prepared.core_tracker_context_available[None], sparse, 0).astype(np.float32))
            predicted = predictor(motion, sparse)
            tracker = torch.from_numpy(prepared.current_tracker_raw[None])
            _, condition, geometry = build_current_realtime_pose_conditions(
                initial_pose_raw=normalizer.inverse_pose(predicted[:, 0]), current_tracker_raw=tracker,
                joint_offsets_parent=torch.from_numpy(rest.rest_local_positions[None]),
                pose_mean=normalizer.pose_mean, pose_scale=normalizer.pose_scale,
                tracker_mean=normalizer.tracker_mean, tracker_scale=normalizer.tracker_std + normalizer.eps,
                config=runtime.ik_inpainting_config)
            inputs = (motion, predicted, tracker, condition.ik_residual, condition.ik_gap, condition.ik_confidence,
                      condition.denoise_strength, condition.constraint_type, torch.randn(1, 144, generator=generator))
            baseline = normal(*inputs)
            original_condition = dit.prepare_conditioning(motion, predicted, geometry, tracker[..., 9] > .5, *inputs[3:8])
            original = diffusion.projected_ddim_sample_loop(dit, (1, 144),
                projection_fn=lambda pose: project_realtime_pose_xstart(pose, tracker, normalizer.pose_mean, normalizer.pose_scale),
                predictor_current=predicted[:, 0], noise=inputs[8], model_kwargs={'prepared_conditioning': original_condition},
                device=torch.device('cpu'), eta=0, clip_denoised=False)['deployed_pred_pose']
            torch.testing.assert_close(baseline, original, atol=1e-4, rtol=1e-3)
            display = post.target_frame(baseline, tracker)
            context = torch.tensor([[0., 0., .1, .18]])
            positions, _ = post.display_geometry(baseline, tracker, context, display)
            boxes = torch.zeros(1, 2, 15); boxes[..., 3:12] = torch.eye(3).flatten()
            boxes[..., 12:] = torch.tensor([.033, .0095, .1325])
            for h in range(2):
                boxes[0, h, :3] = (positions[0, -1, 18+h] + positions[0, -1, 20+h]) * .5
                boxes[0, h, 1] -= post.radii[h, 0] - .005
            # 同一输入分别测停用、普通避碰和从上方接触；保留确定噪声。
            modes = torch.full((1, 2, 2), tick, dtype=torch.long)
            values = inputs + (boxes, modes, context, display)
            expected = sampler(*values)
            if tick == 0:
                torch.testing.assert_close(expected, baseline, atol=0, rtol=0)
            feed = {name: value.numpy() for name, value in zip(SAMPLER_INPUTS + COLLISION_INPUTS, values)}
            begin = time.perf_counter(); actual = session.run(None, feed)[0]; ms = (time.perf_counter()-begin)*1000
            np.testing.assert_allclose(actual, expected.numpy(), atol=1e-4, rtol=1e-3)
            corrected_p, _ = post.display_geometry(expected, tracker, context, display)
            actual_p, _ = post.display_geometry(torch.from_numpy(actual), tracker, context, display)
            position_error = float((corrected_p-actual_p).norm(dim=-1).max())
            assert position_error <= .001, position_error
            before = post.measurement(baseline, tracker, boxes, modes, context, display)[1].item()
            after = post.measurement(expected, tracker, boxes, modes, context, display)[1].item()
            rows.append(dict(inputs={name: dict(shape=list(v.shape), dtype=str(v.dtype), data=v.flatten().tolist())
                        for name, v in feed.items()}, expected=expected.flatten().tolist(), baseline=baseline.flatten().tolist(),
                        onnx_max_abs=float(np.abs(actual-expected.numpy()).max()), position_error_m=position_error,
                        expected_positions=corrected_p[0].tolist(),
                        original_max_abs=float((baseline-original).abs().max()), cpu_ms=ms,
                        violation_before_m=before, violation_after_m=after))
            raw = normalizer.inverse_pose(baseline).numpy()[0]
            runtime._finish_step(prepared, normalizer.inverse_pose(predicted).numpy()[0], raw, raw,
                                 condition.ik_gap.numpy()[0], condition.ik_confidence.numpy()[0], condition.denoise_strength.numpy()[0])
    result = dict(models=session.manifest['models'], fixtures=rows)
    (args.output_dir / 'sampler_fixtures.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps([{k:v for k,v in row.items() if k not in ('inputs','expected','baseline')} for row in rows], indent=2))


if __name__ == '__main__':
    main()
