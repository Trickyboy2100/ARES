"""New research paddle tool; preserve source arm/gripper collision geometry.

This is neither field tray CAD nor a contact-grasp experiment. The old synthetic
welded tray is replaced only in a new asset, never in the historical source.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf
from ares_r.perception.robot_collision import arm_link_transforms

def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.asset/'asset_manifest.json').read_text())
    req=json.loads((a.asset/'industrial_request.json').read_text())
    transform=np.array(meta['body_bases']['right'])@arm_link_transforms(a.urdf,meta['start_q'])['link6']
    offset=np.array(req['attached_object_collision']['T_link6_object'])[:3,3]
    center=transform[:3,:3]@offset+transform[:3,3]
    target=np.eye(4);target[:3,3]=center
    local=np.linalg.inv(transform)@target
    stage=Usd.Stage.CreateNew(str(a.output/'jaka_paddle.usda'))
    source=Usd.Stage.Open(str(a.asset/'jaka_held.usda'));stage.GetRootLayer().TransferContent(source.Flatten())
    stage.RemovePrim('/Robot/Link_6/WeldedTray')
    cube=UsdGeom.Cube.Define(stage,'/Robot/Link_6/ResearchPaddle');cube.CreateSizeAttr(1.)
    scaled=local@np.diag([.03,a.paddle_width,.04,1.])
    cube.AddTransformOp().Set(Gf.Matrix4d(scaled.T.tolist()))
    cube.CreateDisplayColorAttr([(.9,.55,.15)]);UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if a.color:
        material=UsdShade.Material.Define(stage,'/Robot/ResearchArmMaterial')
        shader=UsdShade.Shader.Define(stage,'/Robot/ResearchArmMaterial/Shader');shader.CreateIdAttr('UsdPreviewSurface')
        from pxr import Sdf
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set((.23,.32,.43))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    stage.GetRootLayer().Save()
    out=dict(meta,scope='PARAMETERIZED_CONTACT_PUSH_COMPOSITION_NOT_FIELD_REFERENCE',
        held_offset_link6_m=offset.tolist(),paddle_center_body_m=center.tolist(),
        paddle_dims_m=[.03,a.paddle_width,.04],level_link6_rotation=transform[:3,:3].tolist(),
        source_asset_sha256=hashlib.sha256((a.asset/'jaka_held.usda').read_bytes()).hexdigest(),
        asset_sha256=hashlib.sha256((a.output/'jaka_paddle.usda').read_bytes()).hexdigest(),
        unchanged_arm_gripper_collisions=True,contact_grasp_tested=False)
    out['collision_approximation']='source mesh convex hulls and research paddle OBB, not measured full CAD contact geometry'
    out['tool_inertia_note']='historical approximate merged link6 inertia retained for all study conditions, not calibrated new paddle inertia'
    out['display_material_override_only']=a.color
    (a.output/'asset_manifest.json').write_text(json.dumps(out,indent=2))
    print(center)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['asset','urdf','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--color',action='store_true',help='visual material override only, no physics attribute change')
    p.add_argument('--paddle-width',type=float,default=.06,help='new research tool version; never modify historical assets')
    main(p.parse_args())
