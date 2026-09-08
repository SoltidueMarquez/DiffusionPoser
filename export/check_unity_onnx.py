"""用一段三秒录制检查 ONNX 数值和闭环；不运行全数据集评估。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from export.export_sentis_denoiser import build_arg_parser, load_models, IK_KEYS
from export.unity_onnx_models import SAMPLER_INPUTS
from sample import infer_unity_recording as bridge
from sample.realtime_pose_runtime import RealtimePoseRuntime
from diffusion.realtime_pose_inpainting import build_current_realtime_pose_conditions
from diffusion.realtime_pose_projection import project_realtime_pose_xstart
from utils.parser_util import parse_and_load_from_model


def main(argv=None):
    parser = build_arg_parser()
    parser.add_argument('--input', type=Path, default=Path(
        'sample/assets/unity_recordings/recording_20260829_164358.json'))
    args = parse_and_load_from_model(parser, argv,
        ignore_keys={'normalizer_dir', 'output_dir', 'body_fbx_rest_json', 'input'})
    torch.set_num_threads(1)
    predictor, dit, diffusion, normalizer = load_models(args)
    sessions = []
    for name in ('predictor', 'pose_sampler'):
        path = args.output_dir / f'{name}.onnx'
        onnx.checker.check_model(onnx.load(str(path)))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        sessions.append(ort.InferenceSession(str(path), options,
                                             providers=['CPUExecutionProvider']))
    predictor_session, sampler_session = sessions
    raw = bridge.apply_tracker_availability_overrides(
        bridge.load_unity_tracker_recording(args.input), ignore_hip=True, ignore_feet=False)
    recording = bridge.resample_tracker_recording(raw)
    rest = bridge.load_body_fbx_rest(args.body_fbx_rest_json)
    runtime = RealtimePoseRuntime(predictor, dit, diffusion, torch.device('cpu'),
        rest.rest_local_positions, bridge.rotation_6d_forward_up_np(rest.rest_local_rotations),
        normalizer, **{key: getattr(args, key) for key in IK_KEYS})
    start = bridge.find_first_core_window(recording.available)
    runtime.initialize_history(bridge.build_bootstrap_pose_history(recording, rest, start),
        recording.positions[start:start + 11], recording.rotations_6d[start:start + 11],
        recording.floor_y)
    generator = torch.Generator().manual_seed(10)
    frames, times, samples = [], [], {}
    errors = {'predictor_max_abs': 0.0, 'sampler_max_abs': 0.0}
    previous = None
    checked = 0
    # 保持采样时间轴为 30Hz，只处理选定起点后的三秒。
    end = min(len(recording.times), start + 90)
    with torch.inference_mode():
        for tick, current in enumerate(range(start + 11, end)):
            prepared = runtime._prepare_step(recording.positions[current],
                recording.rotations_6d[current], recording.available[current], recording.floor_y)
            motion = normalizer.normalize_pose(prepared.motion_context[None]).astype(np.float32)
            sparse = normalizer.normalize_predictor_sparse(prepared.core_tracker_context[None])
            sparse = np.where(prepared.core_tracker_context_available[None], sparse, 0).astype(np.float32)
            predicted = predictor_session.run(None, {
                'motion_context': motion, 'core_tracker_context': sparse})[0]
            tracker = torch.from_numpy(prepared.current_tracker_raw[None])
            predicted_tensor = torch.from_numpy(predicted)
            _, condition, geometry = build_current_realtime_pose_conditions(
                initial_pose_raw=normalizer.inverse_pose(predicted_tensor[:, 0]),
                current_tracker_raw=tracker,
                joint_offsets_parent=torch.from_numpy(rest.rest_local_positions[None]),
                pose_mean=normalizer.pose_mean, pose_scale=normalizer.pose_scale,
                tracker_mean=normalizer.tracker_mean,
                tracker_scale=normalizer.tracker_std + normalizer.eps,
                config=runtime.ik_inpainting_config)
            noise = torch.randn(1, 144, generator=generator)
            values = [torch.from_numpy(motion), predicted_tensor, tracker,
                condition.ik_residual, condition.ik_gap, condition.ik_confidence,
                condition.denoise_strength, condition.constraint_type, noise]
            feed = {key: value.numpy() for key, value in zip(SAMPLER_INPUTS, values)}
            deployed = sampler_session.run(None, feed)[0]
            if 30 <= tick < 40:
                # 两个后端接收完全相同的条件；不是比较两条已分岔的历史。
                expected_predictor = predictor(torch.from_numpy(motion), torch.from_numpy(sparse)).numpy()
                prepared_condition = dit.prepare_conditioning(values[0], values[1], geometry,
                    tracker[..., 9] > .5, *values[3:8])
                expected = diffusion.projected_ddim_sample_loop(dit, (1, 144),
                    projection_fn=lambda pose: project_realtime_pose_xstart(
                        pose, tracker, normalizer.pose_mean, normalizer.pose_scale),
                    predictor_current=predicted_tensor[:, 0], noise=noise,
                    model_kwargs={'prepared_conditioning': prepared_condition},
                    device=torch.device('cpu'), eta=0, clip_denoised=False)['deployed_pred_pose'].numpy()
                for key, actual, reference in (
                    ('predictor_max_abs', predicted, expected_predictor),
                    ('sampler_max_abs', deployed, expected)):
                    errors[key] = max(errors[key], float(np.max(np.abs(actual - reference))))
                    np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-3)
                for key, value in feed.items():
                    samples.setdefault(key, []).append(value)
                samples.setdefault('core_tracker_context', []).append(sparse)
                samples.setdefault('expected_predictor', []).append(expected_predictor)
                samples.setdefault('expected_deployed_pose', []).append(expected)
                checked += 1
            raw_pose = normalizer.inverse_pose(torch.from_numpy(deployed)).numpy()[0]
            # 原 runtime 的解码和反馈保持不变，闭环历史只接收 ONNX 部署输出。
            result = runtime._finish_step(prepared,
                normalizer.inverse_pose(predicted_tensor).numpy()[0], raw_pose, raw_pose,
                condition.ik_gap.numpy()[0], condition.ik_confidence.numpy()[0],
                condition.denoise_strength.numpy()[0])
            previous = bridge.resolved_pose_to_unity_frame(result.resolved_pose, rest, previous)
            for value in (previous.root_position, previous.root_rotation_xyzw,
                          previous.pelvis_local_position, previous.local_rotations_xyzw):
                if not np.isfinite(value).all():
                    raise ValueError(f'第 {tick} 帧包含非有限姿态。')
            if tick >= 30:
                frames.append(previous)
                times.append(recording.times[current])
    if checked != 10:
        raise ValueError(f'录制不足以覆盖预热后 10 个 tick：{checked}')
    np.savez(args.output_dir / 'reference_inputs.npz',
             **{key: np.stack(value) for key, value in samples.items()})
    bridge.write_unity_pose_result(args.output_dir / 'onnx_pose_result.json',
                                  np.asarray(times) - times[0], frames)
    report = {**errors, 'checked_ticks': checked, 'output_frames': len(frames),
              'atol': 1e-4, 'rtol': 1e-3, 'provider': 'CPUExecutionProvider'}
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
