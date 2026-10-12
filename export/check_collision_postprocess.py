"""生成 Unity 几何对照夹具，并验证独立后处理 ONNX；不改变离线推理入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from export.residual_collision_postprocess import (
    ResidualCollisionPostprocess, forward_kinematics, load_collision_profile, qrotate,
)


def rest_inputs(module, rest):
    local = torch.tensor(rest['restLocalRotations'], dtype=torch.float32).unsqueeze(0)
    identity = torch.tensor([[0., 0., 0., 1.]])
    positions, rotations = forward_kinematics(torch.zeros(1, 3), identity, module.offsets[:1],
                                             local, module.offsets, module.parents)
    forward = qrotate(rotations, torch.tensor([0., 0., 1.]).expand(1, 24, 3))
    up = qrotate(rotations, torch.tensor([0., 1., 0.]).expand(1, 24, 3))
    pose = torch.cat([forward, up], -1).reshape(1, 144)
    joints = [15, 20, 21, 0, 10, 11]
    tracker = torch.zeros(1, 6, 10)
    origin = positions[:, 15:16] * torch.tensor([1., 0., 1.])
    tracker[..., :3] = positions[:, joints] - origin
    tracker[..., 3:9] = pose.reshape(1, 24, 6)[:, joints]
    tracker[..., 9] = 1
    return pose, tracker


def build_fixtures(module, rest):
    pose, tracker = rest_inputs(module, rest)
    generator = torch.Generator().manual_seed(31)
    fixtures = []
    for case in range(6):
        prior = pose + (torch.randn((1, 24), generator=generator) @ module.directions) * (.02 * case)
        observed = tracker.clone()
        display = module.target_frame(pose, observed)
        context = torch.tensor([[float(display[0, 8]) - .1 + (.08 if case % 2 else 0), .7, .1, .18]])
        positions, _ = module.display_geometry(prior, observed, context, display)
        boxes = torch.zeros(1, 2, 15)
        boxes[..., 3:12] = torch.eye(3).flatten()
        boxes[..., 12:] = torch.tensor([.066, .0095, .1325])
        for h in range(2):
            boxes[0, h, :3] = (positions[0, -1, 18+h] + positions[0, -1, 20+h]) * .5
            boxes[0, h, 1] -= module.radii[h, 0] - .004
        if case == 0:
            boxes[..., 2] += 3
        if case == 4:
            boxes[:, :, :3] = boxes[:, [1, 0], :3]
        modes = torch.full((1, 2, 2), 2 if case % 2 else 1, dtype=torch.long)
        frames = module.interpolated_frames(prior, observed, display)
        positions, rotations = module.display_geometry(prior, observed, context, display)
        residual = torch.zeros_like(prior)
        inputs = (prior, residual, observed, boxes, modes, context, display, torch.tensor(.2))
        before, penetration = module.measurement(prior, observed, boxes, modes, context, display)
        corrected = module(*inputs)
        after, remaining = module.measurement(prior + corrected, observed, boxes, modes, context, display)
        fixtures.append(dict(inputs=inputs, frames=frames, positions=positions[0], rotations=rotations[0],
                             corrected=corrected, metrics=[before.item(), after.item(), penetration.item(), remaining.item()]))
    return fixtures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body_fbx_rest_json', type=Path, required=True)
    parser.add_argument('--collision_profile', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('output/collision_checks'))
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    rest = json.loads(args.body_fbx_rest_json.read_text(encoding='utf-8-sig'))
    profile = load_collision_profile(args.collision_profile, rest)
    module = ResidualCollisionPostprocess(rest, profile, torch.zeros(144), torch.ones(144)).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        fixtures = build_fixtures(module, rest)
        path = args.output_dir / 'collision_postprocess.onnx'
        names = ['prior', 'residual', 'tracker', 'boxes', 'modes', 'context', 'display', 'variance']
        torch.onnx.export(module, fixtures[1]['inputs'], str(path), opset_version=15, dynamo=False,
                          input_names=names, output_names=['corrected_residual'])
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    errors, elapsed = [], []
    for fixture in fixtures:
        feed = {name: value.numpy() for name, value in zip(names, fixture['inputs'])}
        start = time.perf_counter()
        actual = session.run(None, feed)[0]
        elapsed.append((time.perf_counter() - start) * 1000)
        expected = fixture['corrected'].numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
        errors.append(float(np.max(np.abs(actual - expected))))
    serial = []
    for fixture in fixtures:
        row = {key: value.tolist() if isinstance(value, torch.Tensor) else value
               for key, value in fixture.items() if key != 'inputs'}
        row['inputs'] = {name: value.tolist() for name, value in zip(names, fixture['inputs'])}
        serial.append(row)
    result = dict(rest=rest, profile=profile, fixtures=serial, onnx_max_abs=max(errors),
                  onnx_cpu_ms=elapsed, settings=vars(module.settings))
    (args.output_dir / 'geometry_fixtures.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('onnx_max_abs', 'onnx_cpu_ms')}, indent=2))


if __name__ == '__main__':
    main()
