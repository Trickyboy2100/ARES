"""Exact sphere/OBB distances with a proved KD broadphase; not relaxed collision."""
import numpy as np
from scipy.spatial import cKDTree


def rotation(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def signed(points,radii,box):
    local=(points-np.array(box['pose'][:3]))@rotation(box['pose'][3:])
    delta=abs(local)-np.array(box['dims'])/2
    return np.linalg.norm(np.maximum(delta,0),axis=-1)+np.minimum(np.max(delta,axis=-1),0)-radii


def minimum(tree,points,radii,box):
    center=np.array(box['pose'][:3]);_,i=tree.query(center)
    upper=float(signed(points[i:i+1],radii[i:i+1],box)[0])
    # SDF(p)-r >= ||p-center||-||half||-max_radius. Points outside this
    # ball provably cannot improve the best witnessed signed clearance.
    radius=float(np.linalg.norm(np.array(box['dims'])/2)+radii.max()+upper+1e-9)
    indexes=np.asarray(tree.query_ball_point(center,max(radius,1e-9)),dtype=int)
    if not len(indexes):raise RuntimeError('KD bound lost its witnessed candidate')
    values=signed(points[indexes],radii[indexes],box);i=int(values.argmin())
    return float(values[i]),int(indexes[i])


def global_clearances(centers,radii,boxes):
    flat=np.asarray(centers).reshape(-1,3);radii=np.tile(radii,len(centers));tree=cKDTree(flat)
    values={};indexes={}
    for name,box in boxes.items():values[name],indexes[name]=minimum(tree,flat,radii,box)
    return values,indexes


def per_sample_clearances(spheres,boxes):
    points=spheres[:,:,:3];radii=spheres[:,:,3];trees=[cKDTree(p) for p in points]
    return {name:np.array([minimum(tree,p,r,box)[0] for tree,p,r in zip(trees,points,radii)])
            for name,box in boxes.items()}


def gpu_per_sample_clearances(spheres,boxes):
    """Same full sphere/OBB formula, float64 chunks; no broadphase pruning.

    Planner-side diagnostic only. Independent URDF validation remains CPU/KD.
    """
    import torch
    current=spheres.to(dtype=torch.float64);names=list(boxes);output={}
    for start in range(0,len(names),8):
        keys=names[start:start+8];values=[boxes[k] for k in keys]
        center=torch.tensor([b['pose'][:3] for b in values],device=current.device,dtype=torch.float64)
        rot=torch.tensor(np.array([rotation(b['pose'][3:]) for b in values]),device=current.device,dtype=torch.float64)
        half=torch.tensor([b['dims'] for b in values],device=current.device,dtype=torch.float64)/2
        local=torch.einsum('bnsd,bde->bnse',current[None,:,:,:3]-center[:,None,None,:],rot)
        delta=local.abs()-half[:,None,None,:]
        signed=torch.linalg.vector_norm(delta.clamp_min(0),dim=-1)+delta.max(-1).values.clamp_max(0)-current[None,:,:,3]
        minimum=signed.min(-1).values.detach().cpu().numpy()
        output.update({name:minimum[i] for i,name in enumerate(keys)})
    return output


def independent_validator():
    """In-memory research copy; original ARES-R file stays read-only."""
    from pathlib import Path
    import types
    from ares_r.motion import independent_path_validation as original
    source=Path(original.__file__).read_text()
    begin=source.index('    for name, box in request["compiled_scene"]["cuboids"].items():')
    end=source.index('    minimum_name = ',begin)
    replacement='''    from fast_proxy_collision import global_clearances
    by_object, flat_indexes = global_clearances(sphere_centers, sphere_radii, request["compiled_scene"]["cuboids"])
    limiting_component_by_object = {name: labels[index % len(sphere_radii)] for name,index in flat_indexes.items()}
'''
    source=source[:begin]+replacement+source[end:]
    source=source.replace('independent_cpu_urdf_sphere_cuboid_v1','independent_cpu_urdf_sphere_cuboid_exact_kd_v1')
    module=types.ModuleType('ares_r.motion.independent_path_validation_research')
    module.__package__='ares_r.motion';exec(compile(source,str(original.__file__),'exec'),module.__dict__)
    return module.validate_dense_world,source
