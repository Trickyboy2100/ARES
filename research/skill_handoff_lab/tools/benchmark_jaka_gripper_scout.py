"""Physical articulation/mimic readiness test, NOT a grasp/task success claim.

Uses ARES public URDF unchanged, original mesh collisions and effort limits.
State writes only initialize/reset. Every subsequent sample uses drive targets.
"""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--urdf', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--normalize-urdf', action='store_true')
a = p.parse_args()
a.output = a.output.resolve()
a.urdf = a.urdf.resolve()
a.output.mkdir(parents=True, exist_ok=False)
(a.output / 'source.py').write_bytes(Path(__file__).read_bytes())
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, device=f'cuda:{a.gpu}', enable_cameras=True,
                  kit_args=f'--/renderer/activeGpu={a.gpu} --/renderer/multiGpu/enabled=false')
try:
    import numpy as np
    import torch
    import imageio.v2 as imageio
    import isaaclab.sim as sim
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import Camera, CameraCfg
    import_urdf = a.urdf
    if a.normalize_urdf:
        import xml.etree.ElementTree as ET
        import shutil
        tree = ET.parse(a.urdf)
        root = tree.getroot()
        mapping = {node.attrib['name']: 'eg2_' + node.attrib['name']
                   for node in list(root.findall('link')) + list(root.findall('joint'))
                   if node.attrib['name'][0].isdigit()}
        for node in root.iter():
            for attr in ('name', 'link', 'joint'):
                if node.get(attr) in mapping:
                    node.set(attr, mapping[node.get(attr)])
            if node.tag == 'mesh':
                mesh = (a.urdf.parent / node.attrib['filename']).resolve()
                if mesh.name[0].isdigit():
                    # Isaac importer also derives USD prim names from mesh basenames.
                    copy = a.output / 'import_meshes' / ('eg2_' + mesh.name)
                    copy.parent.mkdir(exist_ok=True)
                    shutil.copy2(mesh, copy)
                    assert hashlib.sha256(mesh.read_bytes()).digest() == hashlib.sha256(copy.read_bytes()).digest()
                    mesh = copy.resolve()
                node.set('filename', str(mesh))
        for index, node in enumerate(root.iter('material')):
            if not node.get('name'):
                node.set('name', f'research_material_{index}')
        removed = []
        for node in list(root):
            if node.tag in ('gazebo', 'ros2_control', 'transmission'):
                removed.append(node.tag)
                root.remove(node)
        import_urdf = a.output / 'normalized_import.urdf'
        tree.write(import_urdf, encoding='utf-8', xml_declaration=True)
        (a.output / 'import_normalization.json').write_text(json.dumps({
            'identifier_mapping': mapping, 'removed_nonphysics_ros_extensions': removed,
            'preserved': 'all joint origins, axes, limits, mimic ratios, mesh collisions and inertial elements',
            'source_sha256': hashlib.sha256(a.urdf.read_bytes()).hexdigest(),
            'import_sha256': hashlib.sha256(import_urdf.read_bytes()).hexdigest()}, indent=2))
    context = sim.SimulationContext(sim.SimulationCfg(dt=1/120, device=f'cuda:{a.gpu}'))
    converter = UrdfConverter(UrdfConverterCfg(
        asset_path=str(import_urdf), usd_dir=str(a.output / 'converted'), usd_file_name='jaka_eg2.usd',
        fix_base=True, merge_fixed_joints=True, self_collision=True,
        convert_mimic_joints_to_normal_joints=False, collider_type='convex_decomposition',
        joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0))))
    robot = Articulation(ArticulationCfg(prim_path='/World/Robot',
        spawn=sim.UsdFileCfg(usd_path=converter.usd_path, activate_contact_sensors=True,
              articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=32, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.1), joint_pos={'.*': 0.0}),
        actuators={'arm': ImplicitActuatorCfg(joint_names_expr=['joint[1-6]'], stiffness=400., damping=40.,
                                             effort_limit_sim=100., velocity_limit_sim=1.57),
                   'gripper': ImplicitActuatorCfg(joint_names_expr=['.*4C2_Joint1'], stiffness=20., damping=1.,
                                                 effort_limit_sim=1., velocity_limit_sim=1.)} ))
    # Local primitive floor avoids a network-only GroundPlane USD dependency.
    floor = sim.CuboidCfg(size=(4.,4.,.05), collision_props=sim.CollisionPropertiesCfg(),
                         visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.22,.25,.28)))
    floor.func('/World/Ground', floor, translation=(0.,0.,-.025))
    sim.DomeLightCfg(intensity=1800.).func('/World/Light', sim.DomeLightCfg(intensity=1800.))
    camera = Camera(CameraCfg(prim_path='/World/Camera', height=720, width=960,
        data_types=['rgb'], spawn=sim.PinholeCameraCfg(focal_length=24., focus_distance=1.0,
                   horizontal_aperture=30., clipping_range=(.01,100.))))
    context.reset()
    camera.set_world_poses_from_view(torch.tensor([[.8, .8, .7]], device=context.device),
                                     torch.tensor([[0., 0., .3]], device=context.device))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    grip, _ = robot.find_joints('.*4C2_Joint1')
    assert len(grip) == 1
    target = robot.data.default_joint_pos.clone()
    writer = imageio.get_writer(str(a.output / 'actual_actuation.mp4'), fps=30, codec='libx264')
    q, commands = [], []
    for step in range(720):
        # Smooth bounded 0→0.6→0 rad; no post-reset joint/root pose writes.
        angle = .3 * (1 - np.cos(2 * np.pi * step / 720))
        target[:, grip] = angle
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        context.step()
        robot.update(1/120)
        camera.update(1/120)
        q.append(robot.data.joint_pos.cpu().numpy().copy())
        commands.append(angle)
        if step % 4 == 0:
            writer.append_data(camera.data.output['rgb'][0].cpu().numpy()[:, :, :3])
    writer.close()
    q = np.asarray(q)
    np.savez_compressed(a.output / 'physical_joint_trace.npz', q=q, command=commands)
    (a.output / 'manifest.json').write_text(json.dumps({
        'status': 'COMPLETE', 'scope': 'ARES_JAKA_EG2_ACTUATION_ONLY_NOT_GRASP_OR_BENCHMARK_SUCCESS',
        'urdf_sha256': hashlib.sha256(a.urdf.read_bytes()).hexdigest(),
        'joint_names': robot.joint_names, 'physics_steps': 720, 'physics_dt': 1/120,
        'joint_min': q.min(axis=(0,1)).tolist(), 'joint_max': q.max(axis=(0,1)).tolist(),
        'mimic_preserved': True, 'self_collision_enabled': True,
        'source_mesh_collisions': 'convex_decomposition of existing collision meshes, not full CAD safety certificate',
        'limits': 'source URDF effort and velocity retained; not field controller qualification',
        'no_post_reset_state_writes': True}, indent=2))
except BaseException:
    (a.output / 'exception.txt').write_text(traceback.format_exc())
    traceback.print_exc()
    import os
    os._exit(1)
app.app.close(wait_for_replicator=False, skip_cleanup=True)
