"""Appearance-only cameras/materials for ACTUAL ongoing PhysX evaluation.

Never writes robot/object pose, joint state, velocity, mass or collisions.
All viewpoints at a capture share the same actual physics tick. This is not
training, trajectory playback, field geometry, or success-label generation.
"""
import hashlib,json
from pathlib import Path
import numpy as np

def physics_signature(stage):
    h=hashlib.sha256();count=0
    for p in stage.Traverse():
        if not str(p.GetPath()).startswith('/World/envs/'):continue
        for a in p.GetAttributes():
            n=a.GetName()
            if n.startswith(('physics:','physx','xformOp:')) or n in ['points','faceVertexCounts','faceVertexIndices','size','radius','height','axis']:
                h.update((str(p.GetPath())+'|'+n+'|'+str(a.Get())).encode());count+=1
    return {'sha256':h.hexdigest(),'physical_and_geometric_attributes':count}

def configure(env,output):
    import omni.usd,carb
    from pxr import UsdShade,UsdGeom,UsdLux,Gf,Sdf
    stage=omni.usd.get_context().get_stage();before=physics_signature(stage)
    def material(name,color,metal=.0,roughness=.35):
        m=UsdShade.Material.Define(stage,'/World/PublicationMaterials/'+name)
        s=UsdShade.Shader.Define(stage,str(m.GetPath())+'/Surface');s.CreateIdAttr('UsdPreviewSurface')
        s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        s.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metal);s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface');return m
    mats={'arm':material('ArmSlate',(.33,.48,.62),.35,.32),'paddle':material('PaddleSteel',(.8,.85,.9),.6,.25),
          'cube':material('WorkpieceAmber',(.98,.36,.065),.1,.32),'rail':material('GuideTeal',(.025,.45,.43),.25,.27),
          'table':material('WorktableNavy',(.075,.12,.18),.2,.42)}
    bound=[]
    for p in list(stage.Traverse()):
        path=str(p.GetPath())
        if path.startswith('/World/envs/') and path.endswith('/Robot'):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mats['arm'],bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            bound.append({'prim':path,'material':'arm inherited visual binding'})
        if not path.startswith('/World/envs/') or not p.IsA(UsdGeom.Gprim):continue
        key='cube' if '/Workpiece' in path else ('rail' if '/Rail_' in path else ('table' if '/Table' in path else ('paddle' if 'ResearchPaddle' in path else ('arm' if '/Robot/' in path else None))))
        if key:UsdShade.MaterialBindingAPI.Apply(p).Bind(mats[key]);bound.append({'prim':path,'material':key})
    dome=UsdLux.DomeLight(stage.GetPrimAtPath('/World/Light'))
    if dome:dome.GetIntensityAttr().Set(650.);dome.GetColorAttr().Set(Gf.Vec3f(.88,.94,1.))
    sun=UsdLux.DistantLight.Define(stage,'/World/PublicationKeyLight');sun.CreateIntensityAttr(2200.);sun.CreateAngleAttr(.6)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-32.,-24.,25.))
    carb.settings.get_settings().set('/rtx/post/aa/op',3)
    after=physics_signature(stage)
    if before!=after:raise RuntimeError('appearance edit changed physical/geometric attributes')
    env.publication_output=Path(output);env.publication_records=[];env.publication_view_counter=0
    (Path(output)/'APPEARANCE_IDENTITY.json').write_text(json.dumps({'before':before,'after':after,'equal':before==after,
        'diff_scope':['camera pose/resolution','UsdPreviewSurface color/metallic/roughness binding','Dome/Distant lighting','render antialiasing'],
        'no_changes':['collisions','mass/inertia','initial states','controller','reward','success conditions'],'material_bindings':bound},indent=2))

def _pose(env,which):
    import torch
    if which=='array':
        middle=env.scene.env_origins.mean(0);eye=middle+torch.tensor([9.0,-10.6,9.6],device=env.device)
        target=middle+torch.tensor([.25,-.15,1.1],device=env.device)
    else:
        index=int(which.split('_')[-1]);origin=env.scene.env_origins[index];center=env.center
        if which.startswith('contact'):
            target=origin+center+torch.tensor([.10,-.015,-.012],device=env.device)
            eye=target+torch.tensor([.22,-.30,.28],device=env.device)
        else:
            target=origin+center+torch.tensor([-.16,.10,-.14],device=env.device)
            eye=target+torch.tensor([.78,-.93,.64],device=env.device)
    env.camera.set_world_poses_from_view(eye[None],target[None])

def capture(env,label,output):
    from PIL import Image
    # Existing capture is every16 control steps. Render detailed key states only,
    # but keep the actual array video cadence and never advance physics here.
    tick=int(env.physics_steps);n=env.publication_view_counter;env.publication_view_counter+=1
    if n%12 and label!='FINAL_ACTUAL_STATE':_pose(env,'array');return
    output=Path(output);raw=output/'publication_raw';raw.mkdir(exist_ok=True)
    for which in ['array','single_1','contact_1','single_3','contact_3']:
        if which!='array' and int(which.split('_')[-1])>=env.num_envs:continue
        _pose(env,which)
        # Renderer-only settling after camera motion. This does not call step().
        for _ in range(3):env.sim.render()
        env.camera.update(env.physics_dt,force_recompute=True)
        p=raw/f'{which}_tick{tick:06d}.png';Image.fromarray(env.camera.data.output['rgb'][0].cpu().numpy()).save(p)
        record={'file':str(p.relative_to(output)),'phase':label,'physics_steps':tick,'view':which,
            'state_source':'CURRENT_ACTUAL_PHYSX_NO_STATE_WRITES','is_training':label=='TRAINING'}
        if which!='array':
            index=int(which.split('_')[-1]);record.update(env.physical_snapshot(index));record['env']=index
        env.publication_records.append(record)
    (output/'PUBLICATION_CAMERA_RECORDS.json').write_text(json.dumps(env.publication_records,indent=2))
    _pose(env,'array')
