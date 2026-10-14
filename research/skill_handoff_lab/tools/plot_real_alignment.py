"""Scientific source overlays only, not a reconstructed scan or task outcome."""
import argparse,json,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from ares_r.perception.robot_collision import arm_link_transforms
from pxr import Usd,UsdGeom

p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True);p.add_argument('--ares-r',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
read=lambda name:json.loads((a.audit/name).read_text());field=read('FIELD_REFERENCE.json');g=read('TRAY_GEOMETRY.json');f=read('FRAME_ALIGNMENT.json')
m=PinnedRobotModel(a.ares,a.ares_r);state=field['snapshot']['robot_state'];obstacles=field['snapshot']['environment']['observation']['obstacles']
corners=np.array(list(itertools.product([-1,1],repeat=3)));edges=[(i,j) for i in range(8) for j in range(i) if np.count_nonzero(corners[i]!=corners[j])==1]
def wire(ax,center,dims,rotation=np.eye(3),color='grey',alpha=.5):
    vertices=corners*np.array(dims)/2@rotation.T+np.array(center)
    for i,j in edges:ax.plot(*vertices[[i,j]].T,color=color,alpha=alpha,lw=.6)
def skeleton(ax):
    for side,color in [('left','tab:blue'),('right','tab:red')]:
        q=state[f'{side}_joints_rad'];frames=arm_link_transforms(m.urdf,q)
        points=np.array([(m.body_bases[side]@frames['base_link' if i==0 else f'link{i}'])[:3,3] for i in range(7)])
        ax.plot(*points.T,'o-',c=color,lw=2,ms=3,label=f'{side}: Sep21 measured q / historical FK')
    ax.scatter(0,0,0,c='black',s=20,label='BODY origin')
fig=plt.figure(figsize=(14,6));ax=fig.add_subplot(121,projection='3d');skeleton(ax)
for obstacle in obstacles:
    identity=obstacle['object_id'];color='grey' if identity.startswith('residual') else ('tab:green' if 'table' in identity else 'tab:blue')
    q=obstacle['pose']['quaternion_wxyz'];r=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
    wire(ax,obstacle['pose']['xyz_m'],obstacle['dimensions_m'],r,color,.3)
target=np.array(field['october_target_reported_body_m']);preland=np.array(field['october_preland_reported_body_m'])
ax.scatter(*target,c='orange',marker='*',s=110,label='Oct target: secondary reported, not raw-verified')
ax.scatter(*preland,c='purple',marker='^',s=70,label='Oct preland: secondary reported');ax.plot(*np.array([target,preland]).T,c='purple',ls='--',label='Reported +70mm Z offset')
ax.set_title('PROXY_ONLY: Sep frozen world + reported Oct target\nNOT an October scan reconstruction');ax.legend(fontsize=7,loc='upper left')
bx=fig.add_subplot(122,projection='3d');skeleton(bx)
wire(bx,[.58475,-.48,1.45],[.16,.32,.24],color='tab:cyan')
bx.scatter(.45475,-.48,1.45,c='tab:cyan',s=60,label='3c20dcd held-center research point')
bx.scatter(*target,c='orange',marker='*',s=110,label='Reported October target')
bx.scatter(*preland,c='purple',marker='^',s=70,label='Reported October preland')
bx.quiver(.50475,-.48,1.45,.11,0,0,color='tab:cyan',label='Old PPO +X insertion')
bx.set_title('Task mismatch in BODY coordinates\nResearch cabinet is NOT the field cabinet');bx.legend(fontsize=7,loc='upper left')
for plot in [ax,bx]:
    plot.set_xlabel('BODY X (m)');plot.set_ylabel('BODY Y (m)');plot.set_zlabel('BODY Z (m)');plot.set_xlim(-.6,1.1);plot.set_ylim(-1,1);plot.set_zlim(0,2);plot.set_box_aspect((1.7,2,2));plot.view_init(24,-55)
fig.tight_layout();fig.savefig(a.audit/'REAL_SCENE_OVERLAY.png',dpi=160);fig.savefig(a.audit/'TASK_COMPARISON.png',dpi=160);plt.close(fig)

points=np.load(a.audit/'old_tray_geometry.npz')['vertices_world_m'];center=np.array(g['world_bounds']['center_m'])
fig=plt.figure(figsize=(13,5));ax=fig.add_subplot(121,projection='3d');ax.scatter(*(points-center).T,s=2,c='silver',label='Actual old USD mesh vertices, centered for display')
for name,v in g['markers'].items():
    xyz=np.array(v['T_world_frame'])[:3,3]-center;ax.scatter(*xyz,s=45,label=name.split('/')[-1])
ax.set_title('CAD model vs grasp markers (not measured field tray)');ax.legend(fontsize=7);ax.set_xlabel('old WORLD X (m)');ax.set_ylabel('old WORLD Y (m)');ax.set_zlabel('old WORLD Z (m)')
bx=fig.add_subplot(122);bx.text(.05,.92,'Geometry-source comparison',fontsize=14)
rows=[['old metallic mesh','99.5 x 179.5 x 30 mm','CAD / historical'],['PPO welded box','99.5 x 179.5 x 30 mm','research proxy, no ears/cavity'],['Oct attached object','NOT AVAILABLE','UNVERIFIED transform/envelope'],['physical field tray','CAD / measurements missing','NOT proven same object']]
bx.table(cellText=rows,colWidths=[.28,.35,.37],loc='center',cellLoc='left').scale(1,2);bx.axis('off');fig.tight_layout();fig.savefig(a.audit/'GEOMETRY_SOURCES.png',dpi=160);plt.close(fig)

scene=Usd.Stage.Open(str(a.ares/'isaac_sim/simforge/scenes/main.usd'));cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
components={}
for prim in Usd.PrimRange(scene.GetPrimAtPath('/World/Tray')):
    if prim.GetTypeName() not in ['Xform','Mesh','Cube']:continue
    box=cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():continue
    components[str(prim.GetPath())]={'world_min_m':list(box.GetMin()),'world_max_m':list(box.GetMax()),'world_dims_m':list(box.GetSize())}
(a.audit/'TRAY_COMPONENTS.json').write_text(json.dumps(components,indent=2))
