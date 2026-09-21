"""在 GPU 实际迭代点重算几何/梯度，区分浮点差分误差和轨迹差异。"""
import argparse
import json
from pathlib import Path

import torch

from diffusion.realtime_pose_projection import project_realtime_pose_xstart
from export.residual_collision_postprocess import CollisionSettings, ResidualCollisionPostprocess


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
        assert torch.isfinite(actual).all()
        fixed=list(range(96))+list(range(120,144))
        torch.testing.assert_close(actual[:,fixed],v['residual'][:,fixed],atol=0,rtol=0)
        degenerate='degenerate' in fixture.get('case','')
        project=lambda r:project_realtime_pose_xstart(v['prior']+r,v['tracker'],post.pose_mean,post.pose_scale)
        a,b=project(actual),project(expected)
        torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-3)
        p,_=post.display_geometry(v['prior']+actual,v['tracker'],v['context'],v['display'])
        q,_=post.display_geometry(v['prior']+expected,v['tracker'],v['context'],v['display'])
        final_position=float((p-q).norm(dim=-1).max());assert final_position<=.001
        geometry_error=gradient_error=0.;roundoff_ratio=0.;cosines=[]
        if fixture['active'] and not degenerate:
            trace=torch.tensor(json.loads((args.unity_dir/f'shader_trace_{index}.json').read_text(encoding='utf-8')))
            cache=post.prepare_geometry(v['prior']+v['residual'],v['tracker'],v['context'],v['display'])
            for k in range(5):
                current=trace[k*100:k*100+24].reshape(1,24)
                candidates=v['prior'][:,96:120]+(current+post.difference_offsets)
                loss,_=post.arm_measurement(candidates,v['tracker'],v['boxes'],v['modes'],cache)
                geometric=((loss[1:13]-loss[13:])/(2*post.settings.difference_step)).reshape(2,6,2).permute(0,2,1).reshape(24)
                gradient=geometric+(current-v['residual'][:,96:120]).flatten()/v['variance'].clamp_min(1e-6)
                gpu=trace[k*100+24:k*100+48];error=(gradient-gpu).abs();gradient_error=max(gradient_error,float(error.max()))
                # 中心差分放大舍入误差；按整个活动梯度的 L2 相对误差和方向检查。
                # 靠近零的单分量不适合相对误差；能量舍入尺度仅作诊断，不作为通过阈值。
                scale=torch.maximum(loss[1:13].abs(),loss[13:].abs()).reshape(2,6,2).permute(0,2,1).reshape(24).clamp_min(1)
                bound=32*torch.finfo(torch.float32).eps*scale/post.settings.difference_step
                roundoff_ratio=max(roundoff_ratio,float((error/bound).max()))
                assert error.norm()<=.035+.025*gradient.norm(), (index,k,float(error.norm()),float(gradient.norm()))
                if gradient.norm()>.35:
                    cosine=float(torch.nn.functional.cosine_similarity(gradient[None],gpu[None]))
                    cosines.append(cosine);assert cosine>.99,(index,k,cosine)
                proposed=current-post.settings.h*(1-.99*k/5)*gpu
                next_current=trace[(k+1)*100:(k+1)*100+24].reshape(1,24)
                torch.testing.assert_close(proposed,next_current,atol=2e-6,rtol=1e-5)
            current=trace[500:524].reshape(1,24)
            e,w,_=post.cached_upper_geometry(v['prior'][:,96:120]+(current+post.difference_offsets),cache)
            observed=trace[600:].reshape(25,4,2,10)
            geometry_error=float(torch.maximum((e-observed[...,:3]).norm(dim=-1),(w-observed[...,3:6]).norm(dim=-1)).max())
            assert geometry_error<=.001,(index,geometry_error)
        rows.append(dict(case=fixture.get('case',f'step_{index}'),final_normalized_max_abs=float((a-b).abs().max()),
            final_position_error_m=final_position,candidate_position_error_m=geometry_error,
            gradient_max_abs=gradient_error,gradient_roundoff_bound_ratio=roundoff_ratio,
            min_gradient_cosine=min(cosines) if cosines else None,degenerate=degenerate))
    assert len(rows)==len(source['fixtures'])
    args.output.write_text(json.dumps(dict(status='passed',fixtures=rows),indent=2),encoding='utf-8')
    print(json.dumps(dict(count=len(rows),max_final_position_m=max(r['final_position_error_m'] for r in rows),
        max_geometry_m=max(r['candidate_position_error_m'] for r in rows),max_gradient_bound_ratio=max(r['gradient_roundoff_bound_ratio'] for r in rows)),indent=2))


if __name__=='__main__':main()
