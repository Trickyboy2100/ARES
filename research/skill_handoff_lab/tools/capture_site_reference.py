"""Isaac5.1 paused static reference capture, never a physical task result."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,'width':1280,'height':960})
try:
    import omni.usd,omni.timeline,omni.replicator.core as rep
    import numpy as np
    from PIL import Image
    if not omni.usd.get_context().open_stage(str(a.scene.resolve())):raise RuntimeError('open_stage failed')
    omni.timeline.get_timeline_interface().stop()
    for _ in range(40):app.update()
    # Modest scene illumination keeps pale CAD and proxy edges readable.
    light=rep.create.light(light_type='dome',intensity=250)
    camera=rep.create.camera(position=(2.1,-2.6,2.3),look_at=(.38,0.,1.05))
    product=rep.create.render_product(camera,(1280,960));annotator=rep.AnnotatorRegistry.get_annotator('rgb');annotator.attach([product])
    for _ in range(8):rep.orchestrator.step(rt_subframes=4,pause_timeline=True,delta_time=0.)
    pixels=np.asarray(annotator.get_data())
    if pixels.size==0 or pixels[:,:,:3].std()<1:raise RuntimeError('empty/uniform image')
    Image.fromarray(pixels).save(a.output/'SITE_REFERENCE_ISAAC51_STATIC.png')
    (a.output/'capture_manifest.json').write_text(json.dumps({'evidence_level':'ISAAC_5_1_STATIC_RENDER_HISTORICAL_PROXY_ONLY',
        'physics_steps':0,'physics_task_completed':False,'october_scene_reconstructed':False,'dome_intensity':250,
        'robot_q':'September snapshot, not October live state','target':'secondary reported October coordinate, UNVERIFIED',
        'camera_position':[2.1,-2.6,2.3],'camera_look_at':[.38,0.,1.05],
        'scene_sha256':hashlib.sha256(a.scene.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
finally:app.close()
