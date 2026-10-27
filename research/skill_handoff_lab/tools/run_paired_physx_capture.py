"""Appearance-only source injection into an exact preserved task; cold process.

Requires a completed source run config and optional exact model/config override.
Only camera size, pre-reset recording hooks and disabled legacy GIF capture are
changed. No simulator stepping, action, reward, reset, or success code is changed.
"""
import argparse,ast,hashlib,importlib.util,json,os,sys,traceback
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=2);p.add_argument('--cases',type=int,nargs='*',default=[])
p.add_argument('--ffmpeg',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--helper-dir',type=Path)
p.add_argument('--controller');p.add_argument('--load-policy',type=Path);p.add_argument('--scene',type=Path);p.add_argument('--engine-kind',default='v6')
p.add_argument('--matrix-overview',action='store_true')
p.add_argument('--training-matrix',action='store_true');p.add_argument('--online-training-steps',type=int,default=0);p.add_argument('--eval-episodes',type=int)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
if a.training_matrix:assert a.online_training_steps==12288 and not a.cases,'Separate approved training rerun must be exactly12288 interactions, no near-case sampling'
else:assert a.online_training_steps==0 and a.cases
base=a.source.read_text();source=base
changes=[("prim_path='/World/Preview',height=720,width=960","prim_path='/World/Preview',height=1080,width=1920"),
         ('        def _get_observations(self):','        def _get_observations(self):\n            paired_video.initial(self)'),
         ('    class Adapter(VecEnv):',f"    import paired_physx_video_hooks as paired_video\n    paired_video.configure(env,a.output,{a.cases!r},{str(a.ffmpeg)!r},{a.label!r},matrix={a.matrix_overview!r},training_matrix={a.training_matrix!r})\n    class Adapter(VecEnv):"),
         ("            self.terminal_obs=self._get_observations()['policy'].clone()","            paired_video.post_step(self,done)\n            self.terminal_obs=self._get_observations()['policy'].clone()"),
         ('torch.manual_seed(a.eval_seed);o=vec.reset();iterations=0','torch.manual_seed(a.eval_seed);o=vec.reset();iterations=0\n    paired_video.initial(env)'),
         ('print(json.dumps(result,default=json_default),flush=True);env.close()','paired_video.finalize(env,a.output)\n    print(json.dumps(result,default=json_default),flush=True);env.close()')]
if a.training_matrix:
    changes.extend([('        def _apply_action(self):','        def _apply_action(self):\n            paired_video.training_physics_sample(self)'),
        ("    if a.controller in ['nominal','goal_feedback','damped','legacy','paired','random','fixed_push']:","    paired_video.bind_optimizer(policy,env)\n    if a.controller in ['nominal','goal_feedback','damped','legacy','paired','random','fixed_push']:"),
        ('policy.learn(total_timesteps=a.steps,callback=Callback());policy.save','policy.learn(total_timesteps=a.steps,callback=Callback());paired_video.unbind_optimizer(policy,env);policy.save')])
for old,new in changes:
    if source.count(old)!=1:raise ValueError(f'Expected one video-only patch anchor: {old}')
    source=source.replace(old,new)
start=source.index('    def capture(label):');end=source.index('    class Callback(',start)
source=source[:start]+'    def capture(label):\n        return  # legacy GIF disabled; actual post-step camera recording is separate\n'+source[end:]
# Capture hook is lexical in run(), but constructor/reset may call _get_dones
# before hooks are configured. Resolve the module at run() entry, no state edits.
source=source.replace('def run(a):','def run(a):\n    import paired_physx_video_hooks as paired_video',1)
render=out/'recorded_task.py';render.write_text(source)
def method_asts(text):
    tree=ast.parse(text);env=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Env')
    return {n.name:n for n in env.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
before_methods=method_asts(base);after_methods=method_asts(source)
unchanged=['_pre_physics_step','_reset_idx','state','physical_snapshot','_get_rewards']
proof={}
for name in unchanged:
    left=ast.dump(before_methods[name],include_attributes=False);right=ast.dump(after_methods[name],include_attributes=False)
    if left!=right:raise ValueError(f'Forbidden physical/policy method change: {name}')
    proof[name]=hashlib.sha256(left.encode()).hexdigest()
for name in ['_get_dones','_get_observations','_apply_action']:
    done=after_methods[name]
    done.body=[n for n in done.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='paired_video')]
    assert ast.dump(before_methods[name],include_attributes=False)==ast.dump(done,include_attributes=False)
module_dir=a.helper_dir or a.source.parent
protected={'recorded_task.py','base_task_source.py','paired_physx_video_hooks.py','publication_render_hooks.py','run_paired_physx_capture.py'}
for helper in sorted(module_dir.glob('*.py')):
    if helper.name not in protected and helper.resolve()!=a.source.resolve() and not helper.name.endswith('_source.py'):
        (out/helper.name).write_bytes(helper.read_bytes())
for name in ['contact_credit_v2.py','contact_hybrid_policy.py','contact_task_frame_v7.py','contact_task_frame_v8.py','contact_geometry_v8.py','contact_masked_policy_v8.py','recontact_waypoint_v8.py','recontact_minkowski_v8.py']:
    candidates=[module_dir/name,Path(__file__).with_name(name)]
    helper=next((q for q in candidates if q.exists()),None)
    if helper:(out/name).write_bytes(helper.read_bytes())
# Completed-run snapshots outrank mutable helper/check-out files.
for snapshot,name in {'boundary_credit_source.py':'contact_credit_v2.py','frame_source.py':'contact_task_frame_v7.py','geometry_source.py':'contact_geometry_v8.py','hybrid_policy_source.py':'contact_hybrid_policy.py'}.items():
    preserved=a.source.parent/snapshot
    if preserved.exists():(out/name).write_bytes(preserved.read_bytes())
for name in ['contact_masked_policy_v8.py','recontact_waypoint_v8.py','recontact_minkowski_v8.py']:
    preserved=a.source.parent/name
    if preserved.exists():(out/name).write_bytes(preserved.read_bytes())
for helper in sorted(a.source.parent.glob('*.py')):
    if helper.name not in protected and not helper.name.endswith('_source.py'):
        (out/helper.name).write_bytes(helper.read_bytes())
original_manifest=a.source.parent/'manifest.json'
if original_manifest.exists():
    expected=json.loads(original_manifest.read_text())
    for name,digest in expected.get('dependency_sha256',{}).items():
        assert (out/name).exists() and sha(out/name)==digest, f'Preserved dependency mismatch: {name}'
    for field,name in {'credit_source_sha256':'contact_credit_v2.py','frame_source_sha256':'contact_task_frame_v7.py','geometry_source_sha256':'contact_geometry_v8.py'}.items():
        if expected.get(field):assert sha(out/name)==expected[field], f'Preserved snapshot mismatch: {name}'
(out/'base_task_source.py').write_bytes(a.source.read_bytes())
cfg=json.loads(a.config.read_text());cfg.update(output=out,gpu=a.gpu,steps=a.online_training_steps,skip_capture=False)
if a.training_matrix:cfg.update(load_policy=None,checkpoint_every=max(16384,cfg.get('checkpoint_every',16384)))
if a.eval_episodes is not None:cfg['eval_episodes']=a.eval_episodes
if a.controller:cfg['controller']=a.controller
if a.load_policy:cfg['load_policy']=a.load_policy
if a.scene:cfg['scenario_file']=a.scene
for key in ['asset','scenario_file','load_policy','legacy_policy']:
    if cfg.get(key):cfg[key]=Path(cfg[key])
args=argparse.Namespace(**cfg)
(out/'VIDEO_SOURCE_PATCH.json').write_text(json.dumps({'base_source_sha256':sha(a.source),'recorded_source_sha256':sha(render),
    'base_run_config_sha256':sha(a.config),'patches':[{'before':x,'after':y} for x,y in changes],
    'legacy_capture_disabled':True,'physical_policy_method_AST_identity':proof,'termination_AST_identical_after_removing_record_only_call':True,
    'observation_AST_identical_after_removing_record_only_initial_capture':True,
    'helper_snapshots_sha256':{p.name:sha(p) for p in out.glob('*.py') if p.name not in ['recorded_task.py','base_task_source.py']},
    'scope':'appearance and recording only; exact policy/control/reward/reset/terminal retained'},indent=2))
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,device=f'cuda:{a.gpu}',enable_cameras=True,kit_args=f'--/renderer/activeGpu={a.gpu} --/renderer/multiGpu/enabled=false')
try:
    sys.path.insert(0,str(out))
    spec=importlib.util.spec_from_file_location('recorded_task',render);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.run(args);(out/'RUN_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','scope':'ACTUAL_PHYSX_ONLINE_TRAINING_VISUAL_RERUN_NOT_FORMAL_PERFORMANCE' if a.training_matrix else 'ACTUAL_PHYSX_CAMERA_EVALUATION_NOT_TRAINING'}))
except BaseException:
    (out/'exception.txt').write_text(traceback.format_exc());traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
launcher.app.close(wait_for_replicator=False,skip_cleanup=True)
