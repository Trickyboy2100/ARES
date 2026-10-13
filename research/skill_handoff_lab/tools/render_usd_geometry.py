"""CPU mesh preview of the existing Isaac USD; not an Isaac renderer or physics run."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pxr import Usd,UsdGeom

def main():
    p=argparse.ArgumentParser(); p.add_argument("--scene",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    stage=Usd.Stage.Open(str(a.scene)); cache=UsdGeom.XformCache()
    fig=plt.figure(figsize=(13,9)); ax=fig.add_subplot(111,projection="3d")
    meshes=[]; bounds=[]
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh): continue
        path=str(prim.GetPath())
        if "/collisions" in path: continue
        mesh=UsdGeom.Mesh(prim); pts=np.asarray(mesh.GetPointsAttr().Get(),float)
        if pts.size==0: continue
        matrix=np.array(cache.GetLocalToWorldTransform(prim))
        xyz=(np.c_[pts,np.ones(len(pts))]@matrix)[:,:3]
        # Preview the industrial robot/workstation region; exclude room shell.
        center=np.mean(xyz,axis=0)
        if np.linalg.norm(center[:2])>5 or np.ptp(xyz,axis=0).max()>5: continue
        counts=mesh.GetFaceVertexCountsAttr().Get(); indices=mesh.GetFaceVertexIndicesAttr().Get()
        faces=[]; offset=0
        stride=1
        for i,n in enumerate(counts):
            if i%stride==0: faces.append(xyz[np.asarray(indices[offset:offset+n],int)])
            offset+=n
        color="#7596bb" if "left" in path else "#c98b69" if "right" in path else "#929c9b"
        ax.add_collection3d(Poly3DCollection(faces,facecolor=color,edgecolor="none",alpha=1.))
        bounds.append(xyz); meshes.append({"path":path,"vertices":len(pts),"faces":len(counts),"preview_stride":stride})
    if not bounds: raise RuntimeError("no USD meshes loaded")
    xyz=np.concatenate(bounds); low=xyz.min(0); high=xyz.max(0)
    ax.set_xlim(low[0],high[0]); ax.set_ylim(low[1],high[1]); ax.set_zlim(max(-.1,low[2]),high[2])
    ax.set_box_aspect(np.maximum(high-low,.1)); ax.view_init(elev=24,azim=-55)
    ax.set_xlabel("USD WORLD X (m)"); ax.set_ylabel("USD WORLD Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("ARES Isaac Sim 5.1 scene assets | CPU mesh preview\nStatic USD geometry; no trajectory or physics success claimed")
    fig.tight_layout(); fig.savefig(a.output/"ares_usd_geometry.png",dpi=160); plt.close(fig)
    manifest={"evidence_level":"STATIC_USD_CPU_GEOMETRY_RENDER","isaac_simulation":False,
        "scene":str(a.scene.resolve()),"scene_sha256":hashlib.sha256(a.scene.read_bytes()).hexdigest(),
        "mesh_count":len(meshes),"meshes":meshes,"world_body_alignment":"NOT_VERIFIED"}
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"mesh_count":len(meshes),"output":str(a.output)}))
if __name__=="__main__": main()
