"""在 GPU 实际迭代点重算几何/梯度，区分浮点差分误差和轨迹差异。"""
import argparse
import json
from pathlib import Path

import torch

from diffusion.realtime_pose_projection import project_realtime_pose_xstart
from export.residual_collision_postprocess import CollisionSettings, ResidualCollisionPostprocess, qrotate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,default=Path('output/collision_compute/shader_fixtures.json'))
    parser.add_argument('--unity_dir',type=Path,default=Path('../FLUIDUnity/Library/CollisionChecks'))
    parser.add_argument('--output',type=Path,default=Path('output/collision_compute/shader_check.json'))
    args=parser.parse_args();torch.set_num_threads(1)
    source=json.loads(args.fixtures.read_text(encoding='utf-8'));m=source['manifest']
    post=ResidualCollisionPostprocess(m,m['profile'],torch.tensor(m['pose_mean']),torch.tensor(m['pose_scale']),CollisionSettings(**m['settings']))
    report=json.loads((args.unity_dir/'compute_report.json').read_text(encoding='utf-8'))
    rows=[]
    for index,(fixture,result) in enumerate(zip(source['fixtures'],report['shader_steps'])):
        v={k:torch.tensor(x,dtype=torch.long if k=='modes' else torch.float32) for k,x in fixture['inputs'].items()}
        actual=torch.tensor(result['actual']).reshape(1,144);expected=torch.tensor(fixture['expected'])
        height_context=v.get('height_context')
        axis=v['context'][:,9:12]*torch.tensor([1.,0.,1.]);axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        previous_forward=v.get('forward_context',torch.zeros(1,1))
        actual_forward=torch.tensor([[result.get('forward',0.)]])
        expected_forward=torch.tensor(fixture.get('expected_forward',[[0.]]))
        assert float((actual_forward-expected_forward).abs().max())<=.0001
        assert float(actual_forward.abs().max())<=post.settings.head_forward_limit+1e-6
        actual_height=torch.tensor([result.get('height',0.)])
        expected_height=torch.tensor(fixture.get('expected_height',[[0.]])).reshape(-1)
        height_error=float((actual_height-expected_height).abs().max())
        assert height_error<=.0001,(index,height_error)
        assert torch.isfinite(actual).all()
        fixed=[i for i in range(144) if i not in post.active_indices.tolist()]
        torch.testing.assert_close(actual[:,fixed],v['residual'][:,fixed],atol=0,rtol=0)
        degenerate='degenerate' in fixture.get('case','')
        project=lambda r:project_realtime_pose_xstart(v['prior']+r,v['tracker'],post.pose_mean,post.pose_scale)
        a,b=project(actual),project(expected)
        # 有限差分的舍入可在近零归一化分量产生大相对误差；验收实际骨架位置与旋转。
        raw_a=(a*post.pose_scale+post.pose_mean).reshape(1,24,2,3)
        raw_b=(b*post.pose_scale+post.pose_mean).reshape(1,24,2,3)
        assert (raw_a-raw_b).norm(dim=-1).max() < .00175  # 约 0.1 度方向误差
        p,_=post.display_geometry(v['prior']+actual,v['tracker'],v['context'],v['display'],actual_height,axis*actual_forward)
        q,_=post.display_geometry(v['prior']+expected,v['tracker'],v['context'],v['display'],expected_height,axis*expected_forward)
        final_position=float((p-q).norm(dim=-1).max());assert final_position<=.001
        geometry_error=gradient_error=0.;roundoff_ratio=0.;cosines=[]
        if fixture['active'] and not degenerate:
            trace=torch.tensor(json.loads((args.unity_dir/f'shader_trace_{index}.json').read_text(encoding='utf-8')))
            history=v.get('correction_context',torch.zeros(1,74))
            base_pose=v['prior']+v['residual']
            hc=height_context if height_context is not None else torch.tensor([[0.,1/30]])
            ph=v.get('initial_height',hc[:,:1]).clamp(-post.settings.head_y_limit,post.settings.head_y_limit)
            pelvis_reference=post.pelvis_reference(base_pose,v['tracker'],v['context'],v['display'],ph,hc,history)
            target=pelvis_reference[1][0];base=target-history[0,73]
            selected=(p[:,-1,0]*pelvis_reference[0]).sum(-1)[0]
            torch.testing.assert_close(trace[1012],base,atol=2e-6,rtol=0)
            torch.testing.assert_close(trace[1013],target,atol=2e-6,rtol=0)
            torch.testing.assert_close(trace[1014],pelvis_reference[2][0],atol=.001,rtol=1e-4)
            torch.testing.assert_close(trace[1015],selected-base,atol=2e-6,rtol=0)
            for k in range(5):
                current=trace[k*146:k*146+72].reshape(1,72)
                candidates=v['prior']+v['residual']+(current+post.difference_offsets-v['residual'][:,post.active_indices])@post.directions
                height=trace[1000+k*2]*.1 if height_context is not None else None
                previous=height_context[:,:1] if height_context is not None else None
                f=trace[1030+k*2]*.1
                options=dict(forward=axis*f,previous_forward=previous_forward)
                loss,_=post.measurement(candidates,v['tracker'],v['boxes'],v['modes'],v['context'],v['display'],height,previous,pelvis_reference,**options)
                if height_context is not None:
                    dy=height+torch.tensor([post.settings.difference_step,-post.settings.difference_step])*.1
                    hy,_=post.measurement(candidates[:1].expand(2,-1),v['tracker'],v['boxes'],v['modes'],v['context'],v['display'],dy,previous,pelvis_reference,**options)
                    hg=(hy[0]-hy[1])/(2*post.settings.difference_step)
                    height_bound=32*torch.finfo(torch.float32).eps*hy.abs().max().clamp_min(1)/post.settings.difference_step
                    assert abs(float(hg-trace[1001+k*2]))<=float(height_bound)+.025*abs(float(hg)),(index,k,float(hg),float(trace[1001+k*2]))
                df=f+torch.tensor([[post.settings.difference_step],[-post.settings.difference_step]])*.1
                fl,_=post.measurement(candidates[:1].expand(2,-1),v['tracker'],v['boxes'],v['modes'],v['context'],v['display'],height,previous,pelvis_reference,axis*df,previous_forward)
                fg=(fl[0]-fl[1])/(2*post.settings.difference_step)
                fb=32*torch.finfo(torch.float32).eps*fl.abs().max().clamp_min(1)/post.settings.difference_step
                assert abs(float(fg-trace[1031+k*2]))<=float(fb)+.025*abs(float(fg)),(index,k,'forward gradient')
                geometric=(loss[1:73]-loss[73:])/(2*post.settings.difference_step)
                gradient=geometric+(current-v['residual'][:,post.active_indices]).flatten()/v['variance'].clamp_min(1e-6)
                if "correction_context" in v:
                    history=v["correction_context"];gradient[24:]+=post.settings.correction_smooth_weight*history[0,72]*(current[0,24:]-v["residual"][0,post.active_indices[24:]]-history[0,24:72])
                gpu=trace[k*146+72:k*146+144];error=(gradient-gpu).abs();gradient_error=max(gradient_error,float(error.max()))
                # 中心差分放大舍入误差；按整个活动梯度的 L2 相对误差和方向检查。
                # 靠近零的单分量不适合相对误差；能量舍入尺度仅作诊断，不作为通过阈值。
                scale=torch.maximum(loss[1:73].abs(),loss[73:].abs()).clamp_min(1)
                bound=32*torch.finfo(torch.float32).eps*scale/post.settings.difference_step
                roundoff_ratio=max(roundoff_ratio,float((error/bound).max()))
                assert error.norm()<=bound.norm()+.025*gradient.norm(), (index,k,float(error.norm()),float(gradient.norm()))
                if gradient.norm()>.35:
                    cosine=float(torch.nn.functional.cosine_similarity(gradient[None],gpu[None]))
                    cosines.append(cosine);assert cosine>.99,(index,k,cosine)
                factor=trace[1102+k*3]
                assert float(factor) in (0.,.125,.25,.5,1.)
                assert trace[1101+k*3]<=trace[1100+k*3]+1e-6
                proposed=current-post.settings.h*(1-.99*k/post.settings.steps)*gpu*factor
                next_current=(trace[(k+1)*146:(k+1)*146+72] if k<4 else trace[800:872]).reshape(1,72)
                torch.testing.assert_close(proposed,next_current,atol=2e-6,rtol=1e-5)
                pair=torch.cat([current,next_current])
                pair_pose=v['prior']+v['residual']+(pair-v['residual'][:,post.active_indices])@post.directions
                root,rq,pp,_,_=post.target_components(pair_pose,v['tracker'],v['display'],height,axis*torch.stack([f,(trace[1030+(k+1)*2] if k<4 else trace[1020])*.1])[:,None])
                forward=((root+qrotate(rq,pp))*pelvis_reference[0]).sum(-1)
                fee=pelvis_reference[2][0]*post.settings.pelvis_forward_update_cost*(forward[1]-forward[0]).abs()/(2*post.settings.m**2)
                assert trace[1101+k*3]+fee<=trace[1100+k*3]+.001,(index,k,float(fee))
            current=trace[800:872].reshape(1,72)
            normalized=v['prior']+v['residual']+(current-v['residual'][:,post.active_indices])@post.directions
            final_height=trace[1010]*.1 if height_context is not None else None
            positions,_=post.display_geometry(normalized,v['tracker'],v['context'],v['display'],final_height,axis*trace[1020]*.1)
            observed=trace[2000:].reshape(1,4,24,7)
            geometry_error=float((positions-observed[...,:3]).norm(dim=-1).max())
            assert geometry_error<=.001,(index,geometry_error)
        rows.append(dict(case=fixture.get('case',f'step_{index}'),height_error_m=height_error,final_normalized_max_abs=float((a-b).abs().max()),
            final_position_error_m=final_position,candidate_position_error_m=geometry_error,
            gradient_max_abs=gradient_error,gradient_roundoff_bound_ratio=roundoff_ratio,
            min_gradient_cosine=min(cosines) if cosines else None,degenerate=degenerate))
    assert len(rows)==len(source['fixtures'])
    args.output.write_text(json.dumps(dict(status='passed',fixtures=rows),indent=2),encoding='utf-8')
    print(json.dumps(dict(count=len(rows),max_final_position_m=max(r['final_position_error_m'] for r in rows),
        max_geometry_m=max(r['candidate_position_error_m'] for r in rows),max_gradient_bound_ratio=max(r['gradient_roundoff_bound_ratio'] for r in rows)),indent=2))


if __name__=='__main__':main()
