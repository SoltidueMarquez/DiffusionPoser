"""将现有 NPZ 中三个 tick 转为 Unity 可直接读取的最小迁移参考。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from export.export_sentis_denoiser import build_arg_parser, load_models, IK_KEYS
from sample import infer_unity_recording as bridge
from sample.realtime_pose_runtime import RealtimePoseRuntime, decode_and_resolve_pose
from diffusion.realtime_pose_inpainting import build_current_realtime_pose_conditions
from utils.parser_util import parse_and_load_from_model


def main(argv=None):
    parser = build_arg_parser()
    parser.add_argument('--input', type=Path, default=Path('sample/assets/unity_recordings/recording_20260829_164358.json'))
    args = parse_and_load_from_model(parser, argv, ignore_keys={'normalizer_dir','output_dir','body_fbx_rest_json','input'})
    torch.set_num_threads(1)
    predictor, dit, diffusion, normalizer = load_models(args)
    rest = bridge.load_body_fbx_rest(args.body_fbx_rest_json)
    recording = bridge.resample_tracker_recording(bridge.apply_tracker_availability_overrides(
        bridge.load_unity_tracker_recording(args.input), ignore_hip=True, ignore_feet=False))
    start = bridge.find_first_core_window(recording.available)
    runtime = RealtimePoseRuntime(predictor, dit, diffusion, torch.device('cpu'),
        rest.rest_local_positions, bridge.rotation_6d_forward_up_np(rest.rest_local_rotations), normalizer,
        **{key: getattr(args,key) for key in IK_KEYS})
    runtime.initialize_history(bridge.build_bootstrap_pose_history(recording,rest,start),
        recording.positions[start:start+11],recording.rotations_6d[start:start+11],recording.floor_y)
    prepared = runtime._prepare_step(recording.positions[start+11],recording.rotations_6d[start+11],recording.available[start+11],recording.floor_y)
    payload = {'recording': str(args.input.resolve()), 'initial_motion': normalizer.normalize_pose(prepared.motion_context).tolist(),
        'initial_sparse': normalizer.normalize_predictor_sparse(prepared.core_tracker_context).tolist(),
        'initial_tracker': prepared.current_tracker_raw.tolist(), 'ticks': []}
    reference = np.load(args.output_dir/'reference_inputs.npz')
    with torch.inference_mode():
        for index in range(3):
            # 使用保存的同输入模型参考；IK 独立从同一 Predictor horizon 计算。
            values={key: reference[key][index] for key in reference.files}
            predicted=torch.from_numpy(values['predictor_pose_horizon'])
            tracker=torch.from_numpy(values['current_tracker_raw'])
            _, condition, _ = build_current_realtime_pose_conditions(
                initial_pose_raw=normalizer.inverse_pose(predicted[:,0]),current_tracker_raw=tracker,
                joint_offsets_parent=torch.from_numpy(rest.rest_local_positions[None]),pose_mean=normalizer.pose_mean,
                pose_scale=normalizer.pose_scale,tracker_mean=normalizer.tracker_mean,tracker_scale=normalizer.tracker_std+normalizer.eps,
                config=runtime.ik_inpainting_config)
            frame=start+41+index
            forward=recording.rotations_world[frame,0,:,2]
            yaw=float(np.arctan2(forward[0],forward[2]))
            pose=decode_and_resolve_pose(normalizer.inverse_pose(torch.from_numpy(values['expected_deployed_pose'])).numpy()[0],
                tracker.numpy()[0],yaw,recording.positions[frame,0],recording.floor_y,rest.rest_local_positions,
                bridge.rotation_6d_forward_up_np(rest.rest_local_rotations),previous_root_yaw_world=0)
            unity=bridge.resolved_pose_to_unity_frame(pose,rest,None)
            values.update(ik_residual=condition.ik_residual.numpy(), ik_gap=condition.ik_gap.numpy(),
                ik_confidence=condition.ik_confidence.numpy(),denoise_strength=condition.denoise_strength.numpy(),
                constraint_type=condition.constraint_type.numpy())
            sample={key: value.reshape(-1).tolist() for key,value in values.items()}
            sample.update(frame=frame,rootPosition=unity.root_position.tolist(),rootRotation=unity.root_rotation_xyzw.tolist(),
                pelvisLocalPosition=unity.pelvis_local_position.tolist(),localRotations=unity.local_rotations_xyzw.tolist())
            payload['ticks'].append(sample)
    path=args.output_dir/'unity_sequence_reference.json'
    path.write_text(json.dumps(payload),encoding='utf-8')
    print(path.resolve())


if __name__ == '__main__':
    main()
