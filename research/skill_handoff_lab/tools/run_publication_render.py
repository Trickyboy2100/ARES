"""Controlled render-only patch of the preserved v6 physics task.

Only camera dimensions and capture/appearance hooks differ. All dynamic step,
reset, reward, observations and success methods are byte-identical to source.
"""
import argparse,hashlib,json,sys,os,traceback,importlib.util
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--gpu',type=int,default=2)
p.add_argument('--eval-episodes',type=int,default=64)
a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
source=a.source.read_text();changes=[
    ('prim_path=\'/World/Preview\',height=720,width=960','prim_path=\'/World/Preview\',height=1200,width=1920'),
    ('    class Adapter(VecEnv):','    import publication_render_hooks as publication_render\n    publication_render.configure(env,a.output)\n    class Adapter(VecEnv):'),
    ("        env.sim.render();env.camera.update(env.physics_dt,force_recompute=True)","        publication_render.capture(env,label,a.output)\n        env.sim.render();env.camera.update(env.physics_dt,force_recompute=True)")]
for before,after in changes:
    if source.count(before)!=1:raise ValueError(f'expected exactly one appearance hook: {before}')
    source=source.replace(before,after)
render_source=out/'publication_task.py';render_source.write_text(source)
# v6 reads its own sibling estimator snapshot. This is the exact helper version.
helper=Path(__file__).with_name('contact_credit_v2.py');(out/helper.name).write_bytes(helper.read_bytes())
cfg=json.loads((root/'runs/v6b_residual_seed31_validation/run_config.json').read_text())
cfg.update(asset=root/'runs/contact_handoff_narrow_asset',output=out,gpu=a.gpu,num_envs=16,steps=0,skip_capture=False,
    load_policy=root/'runs/v6b_residual_seed31/policy.zip',scenario_file=root/'runs/contact_handoff_v6b_configs/validation.json',
    log_contact_history=True,withdraw_rule=False,policy_mode='gaussian',action_cost='raw',force_switch_signal=None,rule_recovery=False,
    eval_episodes=a.eval_episodes)
args=argparse.Namespace(**cfg)
(out/'RENDER_PATCH_PROVENANCE.json').write_text(json.dumps({'source':str(a.source),'base_source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
    'render_source_sha256':hashlib.sha256(render_source.read_bytes()).hexdigest(),'patches':[{'before':x,'after':y} for x,y in changes],
    'scope':'SYNTHETIC TASK ACTUAL PHYSX EVALUATION; NOT TRAINING OR FIELD'},indent=2))
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,device=f'cuda:{a.gpu}',enable_cameras=True,
    kit_args=f'--/renderer/activeGpu={a.gpu} --/renderer/multiGpu/enabled=false')
try:
    spec=importlib.util.spec_from_file_location('publication_task',render_source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.run(args);(out/'RUN_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE'}))
except BaseException:
    (out/'exception.txt').write_text(traceback.format_exc());traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
launcher.app.close(wait_for_replicator=False,skip_cleanup=True)
