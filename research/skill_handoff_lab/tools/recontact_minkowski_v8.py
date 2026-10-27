"""Exact rectangular-envelope recontact references, no simulator state writes.

Reduces the conservative circular detour while retaining all external workspace,
rail, joint and physical-contact checks. A waypoint is not a safety certificate.
The original circle implementation and its physical failures remain separate.
"""
import torch
from contact_geometry_v8 import contact_ray_clearance, rectangle_axes


def segment_hits_envelope(start, end, object_yaw, tool_yaw, object_half, tool_half, inflation=.002):
    """Exact planar segment/expanded-Minkowski slab test (relative coordinates)."""
    obj, tool = rectangle_axes(object_yaw), rectangle_axes(tool_yaw)
    normals = torch.cat((obj, tool), 1)
    oh = torch.as_tensor(object_half, device=start.device, dtype=start.dtype)
    th = torch.as_tensor(tool_half, device=start.device, dtype=start.dtype)
    support = (torch.einsum('nui,nvi->nuv', normals, obj).abs() * oh).sum(-1)
    support += (torch.einsum('nui,nvi->nuv', normals, tool).abs() * th).sum(-1) + inflation
    position = (normals * start[:, None]).sum(-1)
    velocity = (normals * (end - start)[:, None]).sum(-1)
    parallel = velocity.abs() < 1e-8
    safe_velocity = torch.where(parallel, torch.ones_like(velocity), velocity)
    first, last = (-support - position) / safe_velocity, (support - position) / safe_velocity
    enter = torch.where(parallel, torch.full_like(first, -torch.inf), torch.minimum(first, last)).amax(-1)
    leave = torch.where(parallel, torch.full_like(first, torch.inf), torch.maximum(first, last)).amin(-1)
    impossible = (parallel & (position.abs() > support)).any(-1)
    return (~impossible) & (enter.clamp_min(0) <= leave.clamp_max(1))


def recontact_waypoint_exact(paddle_xy, object_xy, push_direction_xy, mode, contact_distance,
                            object_yaw, tool_yaw, object_half=(.02, .02), tool_half=(.015, .02),
                            margin=.006, max_angle_step=.05, position_tolerance=.001,
                            angle_tolerance=.03):
    """Same mode/return semantics as recontact_waypoint_v8; actual yaw required."""
    assert margin >= .004 and 0 < max_angle_step <= .05
    direction = push_direction_xy / push_direction_xy.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    fallback = torch.zeros_like(direction); fallback[:, 0] = 1.
    direction = torch.where((push_direction_xy.norm(dim=-1) > 1e-8)[:, None], direction, fallback)
    offset = paddle_xy - object_xy
    radius = offset.norm(dim=-1)
    radial = torch.where((radius > 1e-8)[:, None], offset / radius[:, None].clamp_min(1e-9), -direction)
    angle = torch.atan2(radial[:, 1], radial[:, 0])
    desired = torch.atan2(-direction[:, 1], -direction[:, 0])
    error = torch.atan2(torch.sin(desired - angle), torch.cos(desired - angle))
    next_angle = angle + error.clamp(-max_angle_step, max_angle_step)
    next_radial = torch.stack((next_angle.cos(), next_angle.sin()), -1)
    current_radius = contact_ray_clearance(radial, object_yaw, tool_yaw, object_half, tool_half).squeeze(-1) + margin
    orbit_radius = contact_ray_clearance(next_radial, object_yaw, tool_yaw, object_half, tool_half).squeeze(-1) + margin
    backoff = radial * current_radius[:, None]
    orbit = next_radial * orbit_radius[:, None]
    # Never cut an envelope corner while following successive shell waypoints.
    crossing = segment_hits_envelope(offset, orbit, object_yaw, tool_yaw, object_half, tool_half)
    orbit = torch.where(crossing[:, None], backoff, orbit)
    distance = torch.as_tensor(contact_distance, dtype=paddle_xy.dtype, device=paddle_xy.device).reshape(mode.shape)
    approach = -direction * distance[:, None]
    next_mode = torch.where((mode == 0) & (radius >= current_radius - position_tolerance), 1, mode)
    next_mode = torch.where((mode == 1) & (error.abs() <= angle_tolerance)
                            & (radius >= current_radius - position_tolerance), 2, next_mode)
    ready = (mode >= 2) & ((offset - approach).norm(dim=-1) <= position_tolerance)
    next_mode = torch.where(ready, 3, next_mode)
    relative_target = torch.where((next_mode == 0)[:, None], backoff,
                                 torch.where((next_mode == 1)[:, None], orbit, approach))
    return object_xy + relative_target, next_mode, ready
