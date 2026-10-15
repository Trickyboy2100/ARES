"""True October pointcloud/robot/target overlays, for PRIVATE local review only."""
import argparse
import itertools
import json
import os
from controlled_paths import require_private_location
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ares_r.perception.robot_collision import arm_link_transforms, _world_arm_transform, transform_xyz_rpy


def read(p):
    return json.loads(p.read_text())


def main(root, output, urdf):
    require_private_location(output)
    os.umask(0o077)
    report = read(output / 'FIELD_BINDINGS_AND_REPLAY.json')
    source = root / 'real_aligned_sources'
    world = read(source / 'config_snapshot/site_config_20261014/robot_world.json')
    corners = np.array(list(itertools.product([-1, 1], repeat=3)))
    edges = [(i, j) for i in range(8) for j in range(i) if np.count_nonzero(corners[i] != corners[j]) == 1]
    fig = plt.figure(figsize=(18, 7)); geometry_rows = []
    all_sections = plt.figure(figsize=(18, 6))
    for index, scene in enumerate(sorted(report['scenes'], key=lambda x: x['cloud_id'])):
        sid = scene['scene_id']; folder = source / 'scenes' / sid
        snapshot = read(folder / 'scene/snapshot.json')
        geometry = read(folder / 'whole_robot_geometry.json')['boxes']
        z = np.load(source / 'body_cloud_artifacts' / scene['cloud_id'] / 'body_cloud.npz')
        body = z['points_body_m']; colors = z['colors_rgb']
        own = np.zeros(len(body), dtype=bool)
        # Reproduced geometric ownership mask, not ground-truth segmentation.
        for box in geometry:
            if not box.get('filter_owned'):
                continue
            rotation = np.array(box['rotation_body'])
            local = (body - box['center_body_m']) @ rotation
            own |= np.all(np.abs(local) <= np.array(box['half_extents_m']) + .005, axis=1)
        rng = np.random.default_rng(31)
        keep = np.flatnonzero(np.isfinite(body).all(1) & ~own)
        sample = rng.choice(keep, min(40000, len(keep)), replace=False)
        ax = fig.add_subplot(1, 3, index + 1, projection='3d')
        ax.scatter(*body[sample].T, c=body[sample,2], cmap='viridis', vmin=.5, vmax=1.6,
                   s=.5, alpha=.6, rasterized=True)
        clean = np.load(folder / 'scene/clean_residual.npz')['points_body_m']
        ax.scatter(*clean[::max(1, len(clean)//8000)].T, color='#13a89e', s=.4, alpha=.25)
        axes = {}
        bases = {}
        for side, color in [('left', '#306db9'), ('right', '#d4453f')]:
            audit = read(folder / f'{side}_fk_audit.json')
            base = _world_arm_transform(world, side) @ np.array(audit['T_controller_model'])
            bases[side] = base.tolist()
            fk = arm_link_transforms(urdf, scene['state'][side + '_joints_rad'])
            line = np.array([(base @ fk['base_link' if i == 0 else f'link{i}'])[:3, 3] for i in range(7)])
            ax.plot(*line.T, color=color, lw=2.5, marker='o', ms=3, label=side + ' recorded q')
            tool = scene['tool_readbacks'][side]
            values = tool['pose_mm_rad']
            axes[side] = {'T_body_model': base.tolist(), 'T_link6_tcp': transform_xyz_rpy(np.array(values[:3])*.001, values[3:]).tolist(),
                          'tool_id': tool['tool_id'], 'source': 'bound scene SDK audit; base world from Oct14 reference'}
        candidates = [q for q in report['requests'] if q['scene_id'] == sid]
        preland = next((q['goal']['position_m'] for q in candidates if q.get('goal') and 'position_m' in q['goal']), None)
        # For the split case, use the actual preland-plan FK goal, not the
        # SceneSnapshot's unrelated diagnostic A/B joint-delta goal.
        request_record = next((q for q in candidates if 'preland_plan' in q['request_id']), candidates[0])
        request = read(root / request_record['request_id'] / 'planner_request.json')
        if preland is None:
            tcp = np.array(request['T_body_model']) @ arm_link_transforms(urdf, request['goal_rad'])['link6'] @ np.array(request['T_link6_tcp'])
            preland = tcp[:3, 3].tolist()
        goal = np.array(preland)
        ax.scatter(*goal, color='#9b26b6', marker='*', s=95, label='request Preland')
        for entry in report['independent_replays']:
            if any(q['request_id'] == entry['request_id'] for q in candidates):
                path = read(root / entry['request_id'] / 'planning.json')['tcp_path_body_m']
                ax.plot(*np.array(path).T, color='#f38b00', lw=2, label='archived planner TCP')
        attached = request.get('attached_object_collision')
        object_note = 'NOT_MODELED_IN_REQUEST'
        if attached:
            state_q = request['start_rad']
            t = np.array(request['T_body_model']) @ arm_link_transforms(urdf, state_q)['link6']
            box = attached['link6_aabb']
            vertices = (corners * np.array(box['half_extents_m']) + box['center_m']) @ t[:3, :3].T + t[:3, 3]
            for i, j in edges:
                ax.plot(*vertices[[i, j]].T, color='#e345a0', lw=.8)
            object_note = 'CONSERVATIVE_LINK6_AABB_NOT_PHYSICAL_TRAY_CAD'
        ax.set(xlabel='BODY X (m)', ylabel='BODY Y (m)', zlabel='BODY Z (m)',
               xlim=(-.3, 1.35), ylim=(-.9, .9), zlim=(.2, 1.85))
        ax.set_title(f'Epoch {index+1}: {scene["cloud_id"]}\nSCAN surfaces + recorded robot q', fontsize=10)
        ax.view_init(23, -65); ax.legend(fontsize=6, loc='upper left')
        # Goal-centred depth slice, yaw approximately +X; not an inferred CAD opening.
        yaw = (request.get('orientation_lock') or {}).get('final_yaw_target_rad') or 0.
        rotation = Rotation.from_euler('z', yaw).as_matrix()
        local = (body - goal) @ rotation
        region = (np.abs(local[:, 0]) < .02) & (np.abs(local[:, 1]) < .4) & (np.abs(local[:, 2]) < .4) & ~own
        plane = all_sections.add_subplot(1, 3, index+1)
        plane.scatter(local[region, 1]*1000, local[region, 2]*1000, s=1, color='#339a9c', label='measured surface slab +/-20mm')
        obstacles = snapshot['environment']['observation']['obstacles']
        nearby = []
        for box in obstacles:
            if not box['object_id'].startswith('observed_'):
                continue
            q = box['pose']['quaternion_wxyz']; rot = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
            vertices = (corners * np.array(box['dimensions_m'])/2) @ rot.T + box['pose']['xyz_m']
            transformed = (vertices - goal) @ rotation
            if transformed[:, 0].min() <= .02 and transformed[:, 0].max() >= -.02 and np.linalg.norm(transformed.mean(0)[1:]) < .5:
                nearby.append(box['object_id'])
                for i, j in edges:
                    plane.plot(transformed[[i,j], 1]*1000, transformed[[i,j], 2]*1000, color='#dd9933', lw=.6, alpha=.5)
        plane.scatter(0, 0, color='purple', marker='*', s=80)
        plane.set(xlabel='lateral (mm)', ylabel='vertical (mm)', xlim=(-350, 350), ylim=(-350, 350),
                  title=f'Epoch {index+1}: request goal section\nOrange = conservative box projection')
        plane.grid(alpha=.15); plane.legend(fontsize=7)
        nearest = float(cKDTree(body[~own]).query(goal)[0])
        geometry_rows.append({'scene_id': sid, 'cloud_id': scene['cloud_id'], 'frame_chain': axes,
            'request_goal_body_m': preland, 'ownership_filter': 'OBB 5mm model mask, not semantic ground truth',
            'owned_model_points': int(own.sum()), 'remaining_points': int((~own).sum()),
            'archived_clean_residual_points': len(clean), 'section_points': int(region.sum()),
            'goal_nearest_non_owned_scan_surface_m': nearest, 'nearby_proxy_ids': nearby,
            'tray_representation': object_note, 'certified_opening_width_m': None,
            'certified_free_space': 'NOT_ESTABLISHED; missing scan is UNKNOWN, not free or solid',
            'scene_A_B_goal_warning': 'Use actual request, not diagnostic targets.json'})
    fig.suptitle('CONTROLLED REAL OCTOBER DATA | SCAN + recorded robot state; not CAD or physical success', fontsize=12)
    fig.tight_layout(); fig.savefig(output / 'REAL_OCTOBER_OVERLAY.png', dpi=170); plt.close(fig)
    all_sections.suptitle('CONTROLLED | measured surface sections vs planning proxies | blank = UNKNOWN, not certified free', fontsize=12)
    all_sections.tight_layout(); all_sections.savefig(output / 'NARROW_SPACE_SECTIONS.png', dpi=170); plt.close(all_sections)
    (output / 'FIELD_GEOMETRY_AND_FRAMES.json').write_text(json.dumps(geometry_rows, indent=2))
    # Raw planning results and independent reference limit checks, not fair policy comparisons.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, replay in enumerate(report['independent_replays']):
        points = np.array(read(root / replay['request_id'] / 'planning.json')['trajectory_points_rad'])
        axes[0].plot(np.linspace(0,1,len(points)), np.degrees(points[:,2]), label=f'replay path {i+1}')
    config = read(source / 'config_snapshot/site_config_20261014/jaka_mini2_motion.site.json')
    limit = np.degrees(config['upper_rad'][2]-config['soft_limit_margin_rad'])
    axes[0].axhline(limit, color='red', ls='--', label='Oct14 REFERENCE J3 soft max')
    axes[0].set(xlabel='normalized trajectory time', ylabel='J3 (deg)', title='Archived paths: joint branch qualification'); axes[0].legend(fontsize=7)
    x = np.arange(len(report['independent_replays']))
    axes[1].bar(x, [r['world']['min_clearance_m']*1000 for r in report['independent_replays']])
    axes[1].set(xticks=x, xticklabels=[f'path {i+1}' for i in x], ylabel='modeled clearance (mm)', title='Different scene/attachment constraints: NOT a fair method comparison')
    fig.tight_layout(); fig.savefig(output / 'PLANNING_REPLAY_DIAGNOSIS.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--urdf', type=Path, required=True)
    a = p.parse_args(); main(a.root.resolve(), a.output.resolve(), a.urdf.resolve())
