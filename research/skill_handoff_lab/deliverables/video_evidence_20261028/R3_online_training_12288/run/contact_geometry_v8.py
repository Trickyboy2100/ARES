"""Exact planar Minkowski contact-ray boundary for two oriented rectangles.

This is geometry/control engineering, not an RL novelty or 3D safety proof.
"""
import torch

def rectangle_axes(yaw):
    c,s=yaw.cos(),yaw.sin()
    return torch.stack((torch.stack((c,s),-1),torch.stack((-s,c),-1)),1)

def contact_ray_clearance(direction,object_yaw,tool_yaw,object_half,tool_half,lateral_offset=None):
    obj=rectangle_axes(object_yaw);tool=rectangle_axes(tool_yaw);normals=torch.cat((obj,tool),1)
    oh=torch.as_tensor(object_half,device=direction.device,dtype=direction.dtype)
    th=torch.as_tensor(tool_half,device=direction.device,dtype=direction.dtype)
    support=(torch.einsum('nui,nvi->nuv',normals,obj).abs()*oh).sum(-1)
    support+=(torch.einsum('nui,nvi->nuv',normals,tool).abs()*th).sum(-1)
    projection=(normals*direction[:,None]).sum(-1)
    if lateral_offset is not None:
        lateral=torch.stack((-direction[:,1],direction[:,0]),-1)
        support+=projection.sign()*(normals*lateral[:,None]).sum(-1)*lateral_offset[:,None]
    distance=torch.where(projection.abs()>1e-7,support/projection.abs().clamp_min(1e-7),torch.full_like(support,float('inf')))
    return distance.amin(-1,keepdim=True).clamp_min(.001)

def projected_halfwidth(yaw,direction,half):
    axes=rectangle_axes(yaw);h=torch.as_tensor(half,device=yaw.device,dtype=yaw.dtype)
    return ((axes*direction[:,None]).sum(-1).abs()*h).sum(-1)
