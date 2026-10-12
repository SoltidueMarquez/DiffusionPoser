"""分段 ONNX + Python 几何参考；生成 Shader 逐轮数值检查数据。"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export.residual_collision_postprocess import CollisionSettings, ResidualCollisionPostprocess, qrotate


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
            height_context=torch.from_numpy(feed.get('head_height_context',np.array([[0.,1/30,0.]],np.float32)))
            if tuple(height_context.shape)!=(1,3):raise ValueError("head_height_context requires [dy,dt,df]")
            forward_context=height_context[:,2:3]
            height_context=height_context[:,:2]
            forward=forward_context.clone()
            height=height_context[:,:1].clone()
            corrections=torch.from_numpy(feed.get("correction_context",np.zeros((1,366),np.float32)))
            self.corrections=np.zeros((1,365),np.float32)
            for i, (t, ddim) in enumerate(zip(self.manifest['timesteps'], self.manifest['ddim'])):
                residual = self.sessions['step'].run(None, dict(state=state, timestep=np.array([t], np.int64), **cached))[0]
                values = (torch.from_numpy(prior), torch.from_numpy(residual), tracker, boxes, modes, context, display,
                          torch.tensor(ddim['variance']))
                initial_height=height.clone();initial_forward=forward.clone();forward_result={}
                history=torch.cat([corrections[:,i*72:(i+1)*72],corrections[:,360:361],corrections[:,361+i:362+i]],-1)
                if bypass:
                    edited=residual;height=torch.zeros_like(height);forward=torch.zeros_like(forward)
                elif self.post.settings.head_forward_limit>0 or self.post.settings.head_y_limit>0 or self.post.settings.correction_smooth_weight>0 or self.post.settings.pelvis_forward_smooth_weight>0:
                    edited,height=self.post.forward_with_height(*values,height_context,initial_height,correction_context=history,forward_context=forward_context,initial_forward=initial_forward,forward_result=forward_result)
                    forward=forward_result["offset"]
                    edited=edited.numpy()
                else:
                    edited=self.post(*values).numpy();height=torch.zeros_like(height)
                self.corrections[:,i*72+24:(i+1)*72]=(edited-residual)[:,self.post.active_indices.numpy()[24:]]
                axis=context[:,9:12]*context.new_tensor([1.,0.,1.]);axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-6)
                def pelvis_forward(r,h,f):
                    root,rq,pp,_,_=self.post.target_components(torch.from_numpy(prior+r),tracker,display,h,axis*f)
                    return ((root+qrotate(rq,pp))*axis).sum(-1)
                self.corrections[:,360+i]=(pelvis_forward(edited,height,forward)-pelvis_forward(residual,initial_height,torch.zeros_like(initial_forward))).numpy()
                self.steps.append(dict(step=i, state=state.tolist(), inputs={k:v.tolist() for k,v in zip(
                    ['prior', 'residual', 'tracker', 'boxes', 'modes', 'context', 'display', 'variance'], values)},
                    expected=edited.tolist()))
                if self.post.settings.head_forward_limit>0 or self.post.settings.head_y_limit>0 or self.post.settings.correction_smooth_weight>0 or self.post.settings.pelvis_forward_smooth_weight>0:
                    self.steps[-1]['inputs'].update(height_context=height_context.tolist(),initial_height=initial_height.tolist(),correction_context=history.tolist(),forward_context=forward_context.tolist(),initial_forward=initial_forward.tolist())
                    self.steps[-1]['expected_height']=height.tolist()
                    self.steps[-1]['expected_forward']=forward.tolist()
                if i < len(self.manifest['timesteps'])-1:
                    eps = (np.float32(ddim['recip'])*state-edited)/np.float32(ddim['recipm1'])
                    state = edited*np.float32(ddim['sqrt_alpha_prev'])+np.float32(ddim['sqrt_one_minus_alpha_prev'])*eps
            self.forward_offset=forward.numpy()
            self.height_offset=height.numpy()
            return self.sessions['projection'].run(None, dict(predictor_pose_horizon=feed['predictor_pose_horizon'],
                edited_residual=edited, current_tracker_raw=feed['current_tracker_raw']))


def trace_reference(post, row):
    """记录候选几何及梯度；与部署 forward 独立执行，检查每一次更新。"""
    source = row['inputs']
    prior, residual, tracker, boxes, modes, context, display, variance = [torch.tensor(source[k],
        dtype=torch.long if k == 'modes' else torch.float32) for k in
        ['prior', 'residual', 'tracker', 'boxes', 'modes', 'context', 'display', 'variance']]
    trace = torch.zeros(2672)
    hc=torch.tensor(source.get('height_context',[[0.,1/30]]))
    initial=torch.tensor(source.get('initial_height',hc[:,:1].tolist()))
    fc=torch.tensor(source.get("forward_context",[[0.]]));fi=torch.tensor(source.get("initial_forward",fc.tolist()));fr={}
    expected,height=post.forward_with_height(prior,residual,tracker,boxes,modes,context,display,variance,hc,initial,trace,
        torch.tensor(source['correction_context']) if 'correction_context' in source else None,fc,fi,fr)
    axis=context[:,9:12]*torch.tensor([1.,0.,1.]);axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    _,penetration=post.measurement(prior+residual,tracker,boxes,modes,context,display,initial,hc[:,:1],forward=axis*fi,previous_forward=fc)
    row['active']=bool(penetration[0]>0);row['trace']=trace.tolist()
    torch.testing.assert_close(expected,torch.tensor(row['expected']),atol=1e-5,rtol=1e-4)
    row['expected_height']=height.tolist()
    row['expected_forward']=fr['offset'].tolist()


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
