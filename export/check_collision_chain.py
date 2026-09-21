"""分段 ONNX + Python 几何参考；生成 Shader 逐轮数值检查数据。"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export.residual_collision_postprocess import CollisionSettings, ResidualCollisionPostprocess


class CollisionChainReference:
    def __init__(self, directory, post):
        self.manifest = json.loads((directory / 'collision_compute.json').read_text(encoding='utf-8'))
        options = ort.SessionOptions(); options.intra_op_num_threads = 1
        # 此工具验证导出算子，不把 ORT 的融合/近似优化混入差分敏感的参考链。
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.sessions = {k: ort.InferenceSession(str(directory / v), options, providers=['CPUExecutionProvider'])
                         for k, v in self.manifest['models'].items()}
        self.post = post
        self.steps = []

    def run(self, _, feed, bypass=False):
        condition = self.sessions['condition']
        cache = condition.run(None, {i.name: feed[i.name] for i in condition.get_inputs()})
        cached = {o.name: value for o, value in zip(condition.get_outputs(), cache)}
        state = feed['noise'].copy(); prior = feed['predictor_pose_horizon'][:, 0]
        tracker, boxes, modes, context, display = [torch.from_numpy(feed[k]) for k in
            ('current_tracker_raw', 'armrest_geometry', 'contact_mode', 'seated_context', 'display_from_pose')]
        self.steps = []
        with torch.no_grad():
            for i, (t, ddim) in enumerate(zip(self.manifest['timesteps'], self.manifest['ddim'])):
                residual = self.sessions['step'].run(None, dict(state=state, timestep=np.array([t], np.int64), **cached))[0]
                values = (torch.from_numpy(prior), torch.from_numpy(residual), tracker, boxes, modes, context, display,
                          torch.tensor(ddim['variance']))
                edited = residual if bypass else self.post(*values).numpy()
                self.steps.append(dict(step=i, state=state.tolist(), inputs={k:v.tolist() for k,v in zip(
                    ['prior', 'residual', 'tracker', 'boxes', 'modes', 'context', 'display', 'variance'], values)},
                    expected=edited.tolist()))
                if i < len(self.manifest['timesteps'])-1:
                    eps = (np.float32(ddim['recip'])*state-edited)/np.float32(ddim['recipm1'])
                    state = edited*np.float32(ddim['sqrt_alpha_prev'])+np.float32(ddim['sqrt_one_minus_alpha_prev'])*eps
            return self.sessions['projection'].run(None, dict(predictor_pose_horizon=feed['predictor_pose_horizon'],
                edited_residual=edited, current_tracker_raw=feed['current_tracker_raw']))


def trace_reference(post, row):
    """记录候选几何及梯度；与部署 forward 独立执行，检查每一次更新。"""
    source = row['inputs']
    prior, residual, tracker, boxes, modes, context, display, variance = [torch.tensor(source[k],
        dtype=torch.long if k == 'modes' else torch.float32) for k in
        ['prior', 'residual', 'tracker', 'boxes', 'modes', 'context', 'display', 'variance']]
    cache = post.prepare_geometry(prior+residual, tracker, context, display)
    current = reference = residual[:, 96:120]
    trace = torch.zeros(2600); best = current; best_energy = torch.tensor([float('inf')])
    variance = variance.clamp_min(1e-6)
    active = None
    for k in range(post.settings.steps+1):
        candidates = prior[:, 96:120]+(current+post.difference_offsets)
        loss, penetration = post.arm_measurement(candidates, tracker, boxes, modes, cache)
        energy = loss[:1].sum(-1)+(current-reference).square().sum(-1)/(2*variance)
        if k == 0:
            active = bool(penetration[0] > 0)
        if torch.isfinite(energy).all() and energy < best_energy:
            best, best_energy = current.clone(), energy
        if k == post.settings.steps:
            e,w,q = post.cached_upper_geometry(candidates, cache)
            trace[600:] = torch.cat([e,w,q], -1).flatten()
            trace[500:524] = current.flatten();trace[524:548] = (best if active else reference).flatten()
            break
        grad = ((loss[1:13]-loss[13:])/(2*post.settings.difference_step)).reshape(2,6,2).permute(0,2,1).reshape(1,24)
        grad = grad+(current-reference)/variance
        trace[k*100:k*100+24] = current.flatten();trace[k*100+24:k*100+48] = grad.flatten()
        trace[k*100+48] = energy;trace[k*100+49] = best_energy
        proposed = current-post.settings.h*(1-.99*k/post.settings.steps)*grad
        current = proposed if active and torch.isfinite(proposed).all() else current
    row['trace'] = trace.tolist();row['active'] = active
    # 首先确认诊断代码没有与正式参考算法分叉。
    expected = torch.tensor(row['expected'])[:, 96:120]
    torch.testing.assert_close(trace[524:548], expected.flatten(), atol=1e-5, rtol=1e-4)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=Path('output/collision_compute'))
    parser.add_argument('--fixtures', type=Path, default=Path('output/collision_speed/after/sampler_fixtures.json'))
    args=parser.parse_args();torch.set_num_threads(1)
    manifest=json.loads((args.output_dir/'collision_compute.json').read_text(encoding='utf-8'))
    post=ResidualCollisionPostprocess(manifest,manifest['profile'],torch.tensor(manifest['pose_mean']),
        torch.tensor(manifest['pose_scale']),CollisionSettings(**manifest['settings'])).eval()
    chain=CollisionChainReference(args.output_dir,post)
    source=json.loads(args.fixtures.read_text(encoding='utf-8'))
    checks=[];steps=[]
    for bypass in (True,False):
        for index,fixture in enumerate(source['fixtures']):
            feed={k:np.array(v['data'],dtype=v['dtype']).reshape(v['shape']) for k,v in fixture['inputs'].items()}
            actual=chain.run(None,feed,bypass)[0]
            expected=np.array(fixture['baseline' if bypass else 'expected'],np.float32).reshape(1,144)
            np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
            checks.append(dict(bypass=bypass,fixture=index,max_abs=float(np.abs(actual-expected).max())))
            if not bypass:
                for row in chain.steps:
                    with torch.no_grad():trace_reference(post,row)
                    steps.append(row)
    # 独立几何夹具覆盖坐姿、中段接触、交叉扶手和无接触；转换到真实 normalizer。
    geometry_path=Path('output/collision_checks/geometry_fixtures.json')
    if geometry_path.exists():
        geometry=json.loads(geometry_path.read_text(encoding='utf-8'))
        for index, fixture in enumerate(geometry['fixtures']):
            values={k:torch.tensor(v,dtype=torch.long if k=='modes' else torch.float32)
                    for k,v in fixture['inputs'].items()}
            values['prior']=(values['prior']-post.pose_mean)/post.pose_scale
            # 扩展到肘端、腕掌、扶手边缘和从下方接近。
            for location in ('original','elbow','palm','edge','below','degenerate'):
                v={k:t.clone() for k,t in values.items()}
                if location=='degenerate':
                    v['prior']=-post.pose_mean[None]/post.pose_scale;v['tracker'][...,9]=0
                positions,_=post.display_geometry(v['prior'],v['tracker'],v['context'],v['display'])
                if location in ('elbow','palm'):
                    v['boxes'][0,:,:3]=positions[0,-1,[18,19] if location=='elbow' else [20,21]]
                if location=='edge':v['boxes'][...,0]+=.05
                if location=='below':v['boxes'][...,1]+=.1;v['modes'].fill_(1)
                names=['prior','residual','tracker','boxes','modes','context','display','variance']
                with torch.no_grad():expected=post(*(v[k] for k in names))
                row=dict(case=f'geometry_{index}_{location}',step=0,state=torch.zeros(1,144).tolist(),
                    inputs={k:t.tolist() for k,t in v.items()},expected=expected.tolist())
                with torch.no_grad():trace_reference(post,row)
                steps.append(row)
    (args.output_dir/'shader_fixtures.json').write_text(json.dumps(dict(manifest=manifest,fixtures=steps)),encoding='utf-8')
    (args.output_dir/'onnx_check.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
