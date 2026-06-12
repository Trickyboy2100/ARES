#!/usr/bin/env python3
"""Open playground USD, walk all meshes, apply smart materials."""
import omni.usd, time
from pxr import UsdGeom, Gf, Sdf

time.sleep(3)
ctx = omni.usd.get_context()
ctx.open_stage("/home/andyee/Developer/PG-JY/isaac_sim/exported_scenes/lab_playground.usd")
print("opened playground")
time.sleep(1)

stage = ctx.get_stage()

# ========== GROUND + FLOOR ==========
g = stage.GetPrimAtPath("/World/Ground")
if g:
    g.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set([Gf.Vec3f(0.18, 0.20, 0.25)])

fg = UsdGeom.Cube.Define(stage, "/World/FloorBase")
fg.AddScaleOp().Set(Gf.Vec3f(5, 5, 0.002))
fg.AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.01))
fg.GetPrim().CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set([Gf.Vec3f(0.22, 0.25, 0.30)])

# ========== SMART MATERIALS ==========
def classify_and_color(xform):
    for child in xform.GetAllChildren():
        name = child.GetName()
        t = child.GetTypeName()
        name_lower = name.lower()

        if t in ("Xform", "Scope"):
            classify_and_color(child)
        elif t == "Mesh":
            mesh = UsdGeom.Mesh(child)
            ext = mesh.GetExtentAttr().Get()
            if not ext:
                continue
            sx = ext[1][0] - ext[0][0]
            sy = ext[1][1] - ext[0][1]
            sz = ext[1][2] - ext[0][2]
            longest = max(sx, sy, sz)
            vol = sx * sy * sz
            aspect = longest / (min(sx, sy, sz) + 1e-6)

            color = None

            # -- keyword match --
            if "wef" in name_lower or "step" in name_lower or "body" in name_lower:
                color = (0.30, 0.33, 0.40)
            elif "tuolian" in name_lower:
                color = (0.12, 0.13, 0.15)
            elif "gf" in name_lower or "none" in name_lower:
                color = (0.55, 0.57, 0.62) if longest > 0.5 else (0.48, 0.50, 0.55)
            elif "prototype" in name_lower:
                pass  # skip containers
            # -- shape match --
            elif vol > 0.02:
                color = (0.50, 0.42, 0.35)
            elif longest > 0.5 and aspect > 5:
                color = (0.55, 0.57, 0.62)
            elif longest < 0.04:
                color = (0.22, 0.24, 0.28)
            elif 0.1 < longest < 0.4:
                color = (0.42, 0.44, 0.48)
            else:
                color = (0.38, 0.40, 0.44)

            if color:
                child.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set([Gf.Vec3f(*color)])

# ========== APPLY ==========
lab = stage.GetPrimAtPath("/World/jinyu_lab_0526_full_visual_removed")
if not lab:
    for prim in stage.TraverseAll():
        if "jinyu" in prim.GetName().lower():
            lab = prim
            break

if lab:
    classify_and_color(lab)
    print("materials done")
else:
    print("lab not found")
