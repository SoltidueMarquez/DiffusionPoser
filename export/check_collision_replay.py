"""在相同请求快照上比较完整 PyTorch、分段 ORT 和 Shader，排除历史反馈差异。"""
import json
from pathlib import Path

import numpy as np
import torch

from export.export_sentis_denoiser import build_arg_parser, build_collision_postprocess, load_models, COLLISION_KEYS
from export.unity_onnx_models import PoseSampler, SAMPLER_INPUTS
from export.residual_collision_postprocess import COLLISION_INPUTS
from export.check_collision_chain import CollisionChainReference
from utils.parser_util import parse_and_load_from_model


def main():
    parser=build_arg_parser()
    parser.add_argument('--replay_inputs',type=Path,required=True)
    parser.add_argument('--gpu_report',type=Path,required=True)
    parser.add_argument('--fixture_shapes',type=Path,default=Path('output/collision_speed/after/sampler_fixtures.json'))
    args=parse_and_load_from_model(parser,ignore_keys={'normalizer_dir','output_dir','body_fbx_rest_json',
        'sampler_only','sampler_filename','replay_inputs','gpu_report','fixture_shapes'}|COLLISION_KEYS)
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    _,dit,diffusion,normalizer=load_models(args);post,_=build_collision_postprocess(args,normalizer)
    sampler=PoseSampler(dit,diffusion,normalizer,post).eval();chain=CollisionChainReference(args.output_dir,post)
    captured=json.loads(args.replay_inputs.read_text(encoding='utf-8'))
    gpu=json.loads(args.gpu_report.read_text(encoding='utf-8'))['fixtures']
    shapes=json.loads(args.fixture_shapes.read_text(encoding='utf-8'))['fixtures'][0]['inputs']
    assert len(captured)==len(gpu)
    rows=[]
    with torch.no_grad():
        for i,(row,result) in enumerate(zip(captured,gpu)):
            feed={k:np.array(v,dtype=shapes[k]['dtype']).reshape(shapes[k]['shape']) for k,v in row['inputs'].items()}
            values={k:torch.from_numpy(v) for k,v in feed.items()}
            expected=sampler(*(values[k] for k in SAMPLER_INPUTS+COLLISION_INPUTS))
            actual=torch.tensor(result['actual']).reshape(1,144)
            gpu_close=torch.allclose(actual,expected,atol=1e-4,rtol=1e-3)
            ort=chain.run(None,feed)[0]
            ort_close=np.allclose(ort,expected.numpy(),atol=1e-4,rtol=1e-3)
            geometry=lambda x:post.display_geometry(x,values['current_tracker_raw'],values['seated_context'],values['display_from_pose'])[0]
            position_error=float((geometry(actual)-geometry(expected)).norm(dim=-1).max())
            old=torch.tensor(row['expected']).reshape(1,144)
            rows.append(dict(frame=i,gpu_close=gpu_close,ort_close=bool(ort_close),ort_max_abs=float(np.abs(ort-expected.numpy()).max()),shader_max_abs=float((actual-expected).abs().max()),shader_position_error_m=position_error,
                old_gpu_max_abs=float((old-expected).abs().max()),old_gpu_position_error_m=float((geometry(old)-geometry(expected)).norm(dim=-1).max())))
            row['legacy_gpu_expected']=row['expected'];row['expected']=expected.flatten().tolist()
    (args.output_dir/'verified_replay_inputs.json').write_text(json.dumps(captured),encoding='utf-8')
    passed=all(r['gpu_close'] and r['ort_close'] and r['shader_position_error_m']<=.001 for r in rows)
    (args.output_dir/'replay_reference_check.json').write_text(json.dumps(dict(status='passed' if passed else 'failed',frames=rows),indent=2),encoding='utf-8')
    print(json.dumps({k:max(r[k] for r in rows) for k in rows[0] if k!='frame'},indent=2))
    assert passed, '逐帧结果见 replay_reference_check.json'


if __name__=='__main__':main()
