"""Model-grade execution qualification and C2 timing; no hardware interfaces."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from ares_r.perception.robot_collision import arm_link_transforms
from ares_r.perception.robot_owned_filter import grid_spheres_local
from ares_r.motion.independent_path_validation import validate_dense_world, _bounded_knots, _dense_points
from ares_r.motion.tcp_orientation import validate_level_path


def smooth_timing(points, limits, dt=1/120, utilization=.6):
    """C2 joint spline THROUGH knots, independently rechecked for collisions.

    This is not assumed to preserve straight joint segments. A sampled derivative
    bound with margin is supplemented by exact per-cubic polynomial extrema.
    """
    q=np.asarray(points,float)
    keep=np.r_[True,np.max(abs(np.diff(q,axis=0)),axis=1)>1e-8]
    q=q[keep]
    if len(q)<2:raise ValueError('moving trajectory required')
    s=np.linspace(0,1,len(q));spline=CubicSpline(s,q,bc_type=((1,np.zeros(6)),(1,np.zeros(6))))
    # q'=3a*x^2+2b*x+c; q''=6a*x+2b. Inspect each interval's endpoints
    # and the interior vertex of q', rather than relying on a coarse grid.
    vmax=np.zeros(6);amax=np.zeros(6)
    for i,h in enumerate(np.diff(s)):
        a,b,c=spline.c[:3,i]
        for j in range(6):
            x=[0.,h]
            if abs(a[j])>1e-15 and 0 < -b[j]/(3*a[j]) < h:x.append(-b[j]/(3*a[j]))
            vmax[j]=max(vmax[j],max(abs(3*a[j]*v*v+2*b[j]*v+c[j]) for v in x))
            amax[j]=max(amax[j],abs(2*b[j]),abs(6*a[j]*h+2*b[j]))
    duration=max(float(np.max(vmax/(np.array(limits['max_velocity_rad_s'])*utilization))),
                 float(np.sqrt(np.max(amax/(np.array(limits['max_acceleration_rad_s2'])*utilization)))))
    duration=np.ceil(duration/dt)*dt
    times=np.linspace(0,duration,int(round(duration/dt))+1)
    positions=spline(times/duration);positions[0]=q[0];positions[-1]=q[-1]
    return {'times_s':times,'joints_rad':positions,
            'velocity_rad_s':spline(times/duration,1)/duration,
            'acceleration_rad_s2':spline(times/duration,2)/duration**2,
            'duration_s':duration,'utilization':utilization,'analytic_velocity_bound':vmax/duration,
            'analytic_acceleration_bound':amax/duration**2}


def self_spheres(points,request,urdf,ignore):
    """Independent proxy self collision using the frozen sphere model/ignore map.

    No CAD certification is claimed. Same-link overlapping cells are geometry,
    not self collision; only explicitly ignored link pairs are excluded.
    """
    cell=request['planning_parameters'].get('active_sphere_cell_m',.035)
    local={k:grid_spheres_local(v,cell) for k,v in request['collision_model']['arm_link_boxes'].items()}
    envelope=request.get('execution_tool_envelope')
    if envelope and envelope.get('component_model'):
        from ares_r.manipulation.gripper_component_collision import component_spheres_link6
        local['link6'].extend(component_spheres_link6(envelope['component_model'],request['collision_model'],cell))
    else:local['link6'].extend(grid_spheres_local(envelope['box'] if envelope else request['collision_model']['gripper_max_envelope_link6'],cell))
    if request.get('attached_object_collision'):local['link6'].extend(grid_spheres_local(request['attached_object_collision']['link6_aabb'],cell))
    groups={}
    for link,spheres in local.items():
        groups[link]=[(np.array([s['center'] for s in spheres if abs(s['radius']-radius)<1e-12]),radius)
                      for radius in sorted({s['radius'] for s in spheres})]
    pairs=[(x,y) for i,x in enumerate(groups) for y in list(groups)[i+1:]
           if y not in ignore.get(x,[]) and x not in ignore.get(y,[])]
    dense=_dense_points(_bounded_knots(points,max_joint_step_rad=.003),2)
    minimum=float('inf');limiting=None
    for q in dense:
        frames=arm_link_transforms(urdf,q)
        transformed={link:[(centers@frames[link][:3,:3].T+frames[link][:3,3],radius) for centers,radius in values]
                     for link,values in groups.items()}
        for x,y in pairs:
            for xc,xr in transformed[x]:
                tree=cKDTree(xc)
                for yc,yr in transformed[y]:
                    gap=float(tree.query(yc)[0].min()-xr-yr)
                    if gap<minimum:minimum=gap;limiting=(x,y)
    return {'passed':minimum>0,'min_clearance_m':minimum,'limiting_pair':limiting,
            'dense_samples':len(dense),'representation':'independent frozen proxy spheres, not mesh certification',
            'ignore_map_sha256':hashlib.sha256(json.dumps(ignore,sort_keys=True).encode()).hexdigest()}


def qualify(points,request,urdf,limits,ignore):
    q=np.asarray(points);world=validate_dense_world(q.tolist(),dict(request,robot_yaml_urdf=str(urdf)),subdivisions=2)
    base=np.array(request['T_body_model']);tool=np.array(request['T_link6_tcp'])
    fk=lambda x:base@arm_link_transforms(urdf,x)['link6']@tool
    lock=request.get('orientation_lock') or {}
    orientation=validate_level_path(fk,q,subdivisions=2,tolerance_deg=lock.get('max_error_deg',3.),final_yaw_rad=lock.get('final_yaw_target_rad'))
    self_check=self_spheres(q,request,urdf,ignore)
    lower=np.array(limits['lower_rad'])+limits['soft_limit_margin_rad'];upper=np.array(limits['upper_rad'])-limits['soft_limit_margin_rad']
    margin=float(np.minimum(q-lower,upper-q).min())
    checks={'collision':world['collision_free'],'self_collision':self_check['passed'],
            'attached_object':request.get('attached_object_collision') is not None,'soft_limits':margin>=0,
            'orientation':orientation['passed'],'scene_binding':request['scene_digest']==request['compiled_scene']['digest']}
    return {'independent_validation':checks,'qualified':all(checks.values()),'world':world,'self':self_check,
            'orientation':orientation,'soft_margin_rad':margin,'evidence_level':'PLANNING_PROXY_MODEL_NOT_INDUSTRIAL_CERTIFICATION'}
