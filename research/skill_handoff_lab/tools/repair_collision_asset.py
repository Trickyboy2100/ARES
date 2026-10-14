"""Research-only USD repair: attach collision schemas to actual mesh Gprims."""
import argparse,json,hashlib,shutil
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
stage=Usd.Stage.Open(str(a.source/'jaka_held.usda'));changes=[]
while True:
    instances=[p for p in stage.Traverse() if p.IsInstance()]
    if not instances:break
    for prim in instances:prim.SetInstanceable(False)
for prim in stage.Traverse():
    if '/collisions/' not in str(prim.GetPath()):continue
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
        changes.append(str(prim.GetPath()))
    elif prim.HasAPI(UsdPhysics.CollisionAPI):
        prim.RemoveAPI(UsdPhysics.CollisionAPI);prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
if len(changes)!=7:raise ValueError(f'Expected seven arm collision meshes, got {len(changes)}')
stage.Export(str(a.output/'jaka_held.usda'))
manifest=json.loads((a.source/'asset_manifest.json').read_text());manifest['parent_scene_sha256']=manifest['scene_sha256']
manifest['scene_sha256']=hashlib.sha256((a.output/'jaka_held.usda').read_bytes()).hexdigest()
manifest['collision_schema_repair']={'actual_mesh_colliders':changes,'vertices_and_joint_limits_changed':False,
    'original_unchanged':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(a.output/'asset_manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.copy2(a.source/'industrial_request.json',a.output/'industrial_request.json')
print(json.dumps(manifest['collision_schema_repair']))
