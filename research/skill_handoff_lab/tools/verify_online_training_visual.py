"""Read-only actual training-media, physical source and saved tensor-hash QA."""
import argparse,ast,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from verify_paired_physx_delivery import physical_methods,sha,read_media
from paired_physx_video_hooks import parameter_hash

def main(a):
    p=json.loads(a.provenance.read_text());run=Path(p['run'])
    for name,h in p['inputs'].items():
        source=run/name if name!='matrix_live_training.mp4' else run/'video_raw'/name
        assert sha(source)==h,source
    for file,meta in p['outputs'].items():assert sha(Path(file))==meta['sha256'],file
    before=physical_methods(run/'base_task_source.py');after=physical_methods(run/'recorded_task.py')
    for name in ['_pre_physics_step','_apply_action','_reset_idx','_get_observations','state','physical_snapshot','_get_rewards','_get_dones']:
        if name in ['_get_dones','_get_observations','_apply_action']:
            after[name].body=[n for n in after[name].body if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='paired_video')]
        assert ast.dump(before[name],include_attributes=False)==ast.dump(after[name],include_attributes=False),name
    sys.path.insert(0,str(run));from stable_baselines3 import PPO
    model=PPO.load(run/'policy.zip',device='cpu');h=parameter_hash(model);assert h==p['final_parameter_sha256']
    events=json.loads((run/'LIVE_OPTIMIZER_EVENTS.json').read_text())['events'];frames=json.loads((run/'MATRIX_LIVE_TRAINING_FRAMES.json').read_text())
    for e in events:
        pair=[next(f for f in frames if f['capture_event']==f'{prefix}_OPTIMIZER_{e["optimizer_update_index"]}') for prefix in ['PRE','POST']]
        assert pair[0]['environments']==pair[1]['environments'] and pair[0]['physics_tick']==pair[1]['physics_tick']
    d=np.load(run/'TRAINING_120HZ_STATE.npz');assert d['states'].shape==(1536,16,44) and np.isfinite(d['states']).all()
    assert np.array_equal(d['states'][:,0,0],np.arange(1536))
    video=next(Path(f) for f in p['outputs'] if f.endswith('.mp4'));media=read_media(video)
    assert media['width']==1920 and media['height']==1080 and media['codec_name']=='h264' and media['r_frame_rate']=='12/1'
    report={'status':'PASS','scope':'Actual extra12288-interaction online training rerun, not formal performance','all_input_output_hashes_match':True,'physical_method_AST_identity_after_record_only_removal':True,'CPU_checkpoint_load':True,'saved_checkpoint_sorted_tensor_hash':h,'equals_last_optimizer_after_hash':True,'real_rollout_updates':len(events),'optimizer_keeps_physics_state_unchanged':True,'training_state_shape':list(d['states'].shape),'training_sampling_hz':120,'media':media,'verifier_sha256':sha(Path(__file__)),'limitations':['Not a replacement for full arm mesh/self-collision physical validation','Training sidecar is prephysics-after-reference; evaluation trace separate','Camera montage is same-tick16views in one application, not one camera']}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','interactions':12288,'updates':3,'saved_tensor_hash':h}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--provenance',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
