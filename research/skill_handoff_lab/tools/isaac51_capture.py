"""Isaac Sim 5.1 standalone static capture; invoke with Isaac's python.sh."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--scene",type=Path,required=True)
p.add_argument("--output",type=Path,required=True);a=p.parse_args()
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"renderer":"RayTracedLighting","active_gpu":0,
    "physics_gpu":0,"multi_gpu":False,"width":1280,"height":960})
try:
    import omni.usd
    import omni.replicator.core as rep
    import numpy as np
    from PIL import Image
    if not omni.usd.get_context().open_stage(str(a.scene.resolve())):
        raise RuntimeError("failed to open ARES stage")
    for _ in range(60):app.update()
    camera=rep.create.camera(position=(2.6,-2.5,2.5),look_at=(-.15,.8,1.0))
    product=rep.create.render_product(camera,(1280,960))
    annotator=rep.AnnotatorRegistry.get_annotator("rgb");annotator.attach([product])
    for _ in range(8):rep.orchestrator.step(rt_subframes=4,pause_timeline=True)
    image=np.asarray(annotator.get_data())
    if image.size==0 or float(image[:,:,:3].std())<1:
        raise RuntimeError("empty or uniform capture; inspect renderer log")
    Image.fromarray(image).save(a.output/"isaac51_scene.png")
    (a.output/"capture_manifest.json").write_text(json.dumps({
        "evidence_level":"ISAAC_5_1_STATIC_RENDER","physics_task_completed":False,
        "scene_sha256":hashlib.sha256(a.scene.read_bytes()).hexdigest(),
        "camera_position":[2.6,-2.5,2.5],"camera_look_at":[-.15,.8,1.0],
        "resolution":[1280,960]},indent=2)+"\n")
finally:
    app.close()
