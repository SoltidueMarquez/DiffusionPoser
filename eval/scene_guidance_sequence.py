"""读取仓库内配置，沿完整 Tracker 轨迹独立递推各方法。"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from data_loaders.realtime_pose_kinematics import make_yaw_rotation_np
from diffusion.realtime_pose_inpainting import build_current_realtime_pose_conditions
from eval.scene_guidance import sample_scene_guidance
from eval.scene_guidance_benchmark import seeded_feed, write_csv, write_json
from eval.scene_guidance_setup import create_scene_sampler, guidance_configs, prepare_output_directory, validate_inputs
from sample import infer_unity_recording as bridge
from sample.realtime_pose_runtime import RealtimePoseRuntime
from utils.model_util import load_realtime_pose_predictor


def world_reference(recording, captures, start):
    """从原快照恢复世界盒体，跳过首帧椅子标定前的位置；其余帧验证为静态场景。"""
    if len(captures) < 2 or start + 11 + len(captures) > len(recording.times):
        raise ValueError("回放须含至少两帧，且不能超出 Tracker 序列范围。")
    boxes, tracker_errors = [], []
    for tick, row in enumerate(captures):
        current = start + 11 + tick
        rotation = recording.rotations_world[current, 0]
        yaw = np.arctan2(rotation[0, 2], rotation[2, 2])
        heading = make_yaw_rotation_np(np.asarray([yaw]))[0]
        origin = recording.positions[current, 0].copy()
        origin[1] = recording.floor_y
        box = np.asarray(row['inputs']['armrest_geometry'], dtype=np.float64).reshape(2, 15).copy()
        box[:, :3] = box[:, :3] @ heading.T + origin
        box[:, 3:12] = (box[:, 3:12].reshape(2, 3, 3) @ heading.T).reshape(2, 9)
        boxes.append(box)
        tracker = np.asarray(row['inputs']['current_tracker_raw']).reshape(6, 10)
        active = recording.available[current]
        tracker_errors.append(float(np.abs(tracker[active, :3] - ((recording.positions[current]-origin) @ heading)[active]).max()))
    variation = float(np.ptp(boxes[1:], axis=0).max())
    if variation >= 1e-5 or max(tracker_errors) >= 1e-5:
        raise ValueError("快照与 Tracker 序列不对齐，或扶手不是固定世界盒体。")
    return boxes[1].astype(np.float32), dict(static_world_box_variation=variation,
        snapshot_tracker_position_max_error_m=max(tracker_errors), calibration_frame_excluded=0)


def initialize_runtime(predictor, dit, diffusion, device, rest, normalizer, ik, recording, start):
    runtime = RealtimePoseRuntime(predictor, dit, diffusion, device, rest.rest_local_positions,
        bridge.rotation_6d_forward_up_np(rest.rest_local_rotations), normalizer, **ik)
    runtime.initialize_history(bridge.build_bootstrap_pose_history(recording, rest, start),
        recording.positions[start:start+11], recording.rotations_6d[start:start+11], recording.floor_y,
        tracker_available_history=recording.available[start:start+11])
    return runtime


@torch.no_grad()
def prepare_frame(runtime, recording, current, world_boxes, seed):
    """复用实时链路准备 [1,10,144] 历史、[1,11,144] 预测和 IK 条件。"""
    prepared = runtime._prepare_step(recording.positions[current], recording.rotations_6d[current],
        recording.available[current], recording.floor_y)
    normalizer, device = runtime.normalizer, runtime.device
    motion = torch.as_tensor(normalizer.normalize_pose(prepared.motion_context[None]), device=device)
    sparse = normalizer.normalize_predictor_sparse(prepared.core_tracker_context[None])
    sparse = torch.as_tensor(np.where(prepared.core_tracker_context_available[None], sparse, 0).astype(np.float32), device=device)
    predicted = runtime.predictor_model(motion, sparse)
    tracker = torch.as_tensor(prepared.current_tracker_raw[None], device=device)
    _, condition, _ = build_current_realtime_pose_conditions(
        initial_pose_raw=normalizer.inverse_pose(predicted[:, 0]), current_tracker_raw=tracker,
        joint_offsets_parent=torch.as_tensor(runtime.joint_offsets_parent[None], device=device),
        pose_mean=normalizer.pose_mean.to(device), pose_scale=normalizer.pose_scale.to(device),
        tracker_mean=normalizer.tracker_mean.to(device), tracker_scale=(normalizer.tracker_std+normalizer.eps).to(device),
        config=runtime.ik_inpainting_config)
    # 静态盒体逐帧转换到当前 Head-yaw 坐标，三种方法共享同一世界场景。
    heading = make_yaw_rotation_np(np.asarray([prepared.head_yaw_world]))[0]
    origin = recording.positions[current, 0].copy()
    origin[1] = recording.floor_y
    boxes = world_boxes.copy()
    boxes[:, :3] = (boxes[:, :3]-origin) @ heading
    boxes[:, 3:12] = (boxes[:, 3:12].reshape(2, 3, 3) @ heading).reshape(2, 9)
    feed = dict(motion_context=motion, predictor_pose_horizon=predicted, current_tracker_raw=tracker,
        ik_residual=condition.ik_residual, ik_gap=condition.ik_gap, ik_confidence=condition.ik_confidence,
        denoise_strength=condition.denoise_strength, constraint_type=condition.constraint_type,
        noise=torch.empty(1,144, device=device), armrest_geometry=torch.as_tensor(boxes[None], device=device),
        contact_mode=torch.ones((1,2,2), dtype=torch.long, device=device))
    return prepared, seeded_feed(feed, seed, current), heading, origin


@torch.no_grad()
def finish_frame(runtime, prepared, feed, pose):
    normalizer = runtime.normalizer
    raw = normalizer.inverse_pose(pose).cpu().numpy()[0]
    return runtime._finish_step(prepared, normalizer.inverse_pose(feed['predictor_pose_horizon']).cpu().numpy()[0],
        raw, raw, feed['ik_gap'].cpu().numpy()[0], feed['ik_confidence'].cpu().numpy()[0],
        feed['denoise_strength'].cpu().numpy()[0])


def summarize(frame_rows, timing_rows, configs):
    results = []
    for config in configs:
        frames = [r for r in frame_rows if r['method'] == config.method and r['evaluated']]
        timings = [r for r in timing_rows if r['method'] == config.method]
        ms = [r['sampler_ms'] for r in timings]
        results.append(dict(method=config.method, evaluated_frames=len(frames), timed_calls=len(ms),
            sampler_ms=float(np.mean(ms)), sampler_p95_ms=float(np.percentile(ms,95)),
            tracker_error_mm=float(np.mean([r['tracker_error_mm'] for r in frames])),
            penetrating_frame_percent=float(np.mean([r['penetrating'] for r in frames])*100),
            peak_allocated_mb=max(r['peak_allocated_mb'] for r in timings), failures=0))
    return results


def write_report(directory, rows, metadata):
    columns = [('method','方法'),('sampler_ms','完整采样耗时 ms'),('sampler_p95_ms','P95 ms'),
        ('tracker_error_mm','Tracker 误差 mm'),('penetrating_frame_percent','穿透帧比例 %')]
    labels = dict(none='无修正',postedit='PostEdit 式修正',lm='LM 修正',pigdm='ΠGDM 场景适配')
    table = '<table><tr>'+''.join(f'<th>{label}</th>' for _,label in columns)+'</tr>'
    for row in rows:
        table += '<tr>'+''.join(f'<td>{labels[row[key]] if key=="method" else format(row[key],".2f")}</td>' for key,_ in columns)+'</tr>'
    table += '</table>'
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>完整序列固定配置对照</title>
<style>body{{font:16px/1.65 system-ui;margin:36px;max-width:1100px;color:#172033}}table{{border-collapse:collapse}}td,th{{padding:11px;border:1px solid #ddd;text-align:right}}th{{background:#f1f5f9}}img,video{{width:100%;max-width:1100px}}</style>
<h1>完整序列固定配置对照</h1>
<p>同源坐下/扶手接近构造轨迹，共 {metadata['source_frames']} 帧、{metadata['source_duration_seconds']:.1f} 秒、30 Hz。前 11 帧建立历史，本次连续推理 {metadata['rollout_frames']} 帧；前 {metadata['rollout_warmup']} 个输出用于启动，统计其余 {metadata['evaluated_frames_per_seed']} 帧。短程验证：{metadata['short_run']}。</p>
<p>每种方法独立反馈自身输出到下一帧历史；相同 Tracker、初始历史、对应噪声与世界盒体。观测尺度 {metadata['measurement_scale']*1000:g} mm、安全间隙 {metadata['margin']*1000:g} mm、5 步 DDIM。种子 {metadata['seeds']}，重复计时 {metadata['repeats']} 次，预热 {metadata['warmup']} 次。耗时为完整 sampler，含修正，不含 Predictor/IK、文件写入或绘图。</p>
{table}<p>误差与穿透帧比例按独立输出帧统计，重复计时不重复计入效果指标。原始穿透的判定阈值为 0.001 mm；安全间隙违例另存逐帧记录。录制没有全身姿态真值，Tracker 误差由有效设备位置计算。</p>
<p><a href="summary.csv">汇总 CSV</a> · <a href="frames.csv">逐帧指标</a> · <a href="experiment.json">配置与来源</a></p>
<p>各方法的姿态、世界关节、Tracker 与时间戳保存在对应 method_seed*.npz，可独立用于后续绘图。</p>
</html>'''
    (directory/'report.html').write_text(body, encoding='utf-8')


def run_sequence(args):
    sources = validate_inputs(args)
    directory = prepare_output_directory(args)
    configs = guidance_configs(args)
    sampler, scene, normalizer, diffusion, _ = create_scene_sampler(args)
    device = scene.pose_mean.device
    dit = sampler.dit
    predictor = load_realtime_pose_predictor(args.predictor_model_path,device)
    rest = bridge.load_body_fbx_rest(args.body_fbx_rest_json)
    recording = bridge.resample_tracker_recording(bridge.apply_tracker_availability_overrides(
        bridge.load_unity_tracker_recording(args.recording),ignore_hip=True,ignore_feet=False))
    start = bridge.find_first_core_window(recording.available)
    if not recording.available[start:,:3].all():
        raise ValueError("本实验要求完整有效的 Head/双手 Tracker 序列。")
    captures = json.loads(Path(args.replay_inputs).read_text(encoding='utf-8-sig'))
    world_boxes, checks = world_reference(recording,captures,start)
    ik = json.loads(Path(args.model_path).with_name('args.json').read_text(encoding='utf-8'))
    ik = {key:ik[key] for key in ('fabrik_iterations','ik_direction_only_quality','ik_residual_scale',
        'ik_position_solved_quality','ik_gap_low','ik_gap_high','ik_direction_support','ik_untracked_strength')}
    stop = len(recording.times)
    if args.max_frames:
        stop = min(stop, start + 11 + args.rollout_warmup + args.max_frames)
    evaluated_count = stop - start - 11 - args.rollout_warmup
    if evaluated_count < 1:
        raise ValueError("扣除历史与启动帧后没有可计分帧。")
    metadata = dict(protocol='sequence', arguments=vars(args), short_run=bool(args.max_frames),
        constructed_recording=True, source_frames=len(recording.times), rollout_frames=stop-start-11,
        evaluated_frames_per_seed=evaluated_count, first_frame=start+11,
        evaluated_first_frame=start+11+args.rollout_warmup, source_duration_seconds=float(recording.times[-1]),
        seeds=args.seeds, repeats=args.repeats, warmup=args.warmup, rollout_warmup=args.rollout_warmup,
        configs=[c.to_dict() for c in configs], measurement_scale=args.measurement_scale,margin=args.margin,
        ik=ik,device=torch.cuda.get_device_name(device),torch=torch.__version__,precision='FP32',batch_size=1,
        timesteps=list(reversed(diffusion.timestep_map)),world_boxes=world_boxes.tolist(),data_checks=checks,
        sources=sources)
    write_json(directory/'experiment.json',metadata)
    frame_rows, timing_rows = [], []
    started = time.perf_counter()
    with (directory/'timings.jsonl').open('w',encoding='utf-8') as timing_file, (directory/'frames.jsonl').open('w',encoding='utf-8') as frames_file:
        for seed in args.seeds:
            runtimes = {c.method:initialize_runtime(predictor,dit,diffusion,device,rest,normalizer,ik,recording,start) for c in configs}
            trajectories = {c.method:dict(poses=[],joints_world=[],tracker_world=[],times=[]) for c in configs}
            # 仅在第一遍前统一预热；预热不更新任何方法的历史。
            if seed == args.seeds[0]:
                for config in configs:
                    _, feed, _, _ = prepare_frame(runtimes[config.method],recording,start+11,world_boxes,seed)
                    scene.prepare_execution(feed)
                    for _ in range(args.warmup):
                        sample_scene_guidance(sampler,scene,feed,config,args.measurement_scale)
                metadata['shared_geometry_cache'] = dict(
                    backend='CUDA graphs for geometry forward/VJP/Jacobian',
                    initialization_ms=scene.execution.initialization_ms,
                    frame_preparation_included=True, same_point_forward_reused=True)
                write_json(directory/'experiment.json', metadata)
            for current in range(start+11,stop):
                tick = current-start-11
                evaluated = tick >= args.rollout_warmup
                # 轮换方法顺序以分散缓慢的温度/负载漂移；每个方法仍只消费自己的上一帧。
                offset = tick % len(configs)
                order = configs[offset:] + configs[:offset]
                for config in order:
                    runtime = runtimes[config.method]
                    prepared,feed,heading,origin = prepare_frame(runtime,recording,current,world_boxes,seed)
                    first_pose = None
                    for repeat in range(args.repeats if evaluated else 1):
                        pose,timing = sample_scene_guidance(sampler,scene,feed,config,args.measurement_scale,measure=evaluated)
                        if first_pose is None:
                            first_pose = pose.clone()
                        elif not torch.equal(first_pose,pose):
                            raise ValueError('同帧重复计时输出发生变化')
                        if evaluated:
                            row = dict(method=config.method,seed=seed,frame=current,repeat=repeat,**timing)
                            timing_rows.append(row)
                            timing_file.write(json.dumps(row,allow_nan=False)+'\n')
                    metrics = scene.metrics(first_pose,first_pose,feed)
                    metrics.pop('pose_displacement_mm')
                    frame_row = dict(method=config.method,seed=seed,frame=current,time=float(recording.times[current]),evaluated=evaluated,**metrics)
                    frame_rows.append(frame_row)
                    frames_file.write(json.dumps(frame_row,allow_nan=False)+'\n')
                    # FK、反馈与保存只运行一次；重复计时不会人为推进序列。
                    positions,_ = scene.positions_and_rotations(first_pose,feed['current_tracker_raw'])
                    world = positions[0].detach().cpu().numpy() @ heading.T + origin
                    finish_frame(runtime,prepared,feed,first_pose)
                    trajectory = trajectories[config.method]
                    trajectory['poses'].append(first_pose.cpu().numpy()[0])
                    trajectory['joints_world'].append(world.astype(np.float32))
                    trajectory['tracker_world'].append(recording.positions[current])
                    trajectory['times'].append(recording.times[current])
                if tick%25==0 or current==stop-1:
                    timing_file.flush(); frames_file.flush()
                    print(f'[sequence] seed={seed}, frame={tick+1}/{metadata["rollout_frames"]}, elapsed={(time.perf_counter()-started)/60:.1f} min',flush=True)
            for method,trajectory in trajectories.items():
                np.savez_compressed(directory/f'{method}_seed{seed}.npz',**{key:np.asarray(value) for key,value in trajectory.items()},
                    world_boxes=world_boxes,parents=np.asarray(scene.parents),evaluated_start=args.rollout_warmup)
            write_json(directory/f'completed_seed{seed}.json',summarize(frame_rows,timing_rows,configs))
        rows = summarize(frame_rows,timing_rows,configs)
        write_json(directory/'summary.json',rows)
        write_csv(directory/'summary.csv',rows)
        write_csv(directory/'frames.csv',frame_rows)
        write_report(directory,rows,metadata)
        print(json.dumps(rows,ensure_ascii=False,indent=2),flush=True)
    return rows
