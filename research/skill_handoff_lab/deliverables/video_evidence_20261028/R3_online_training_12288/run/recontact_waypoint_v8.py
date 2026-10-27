"""Bounded physical reference waypoints for planar side recontact.

No simulator state writes, labels, learned models or hidden state. The caller
must retain mode/direction in its observable state and execute references through
the existing constrained controller. This clears the workpiece's conservative
planar envelope only: rails, workspace, robot collision and force still require
the caller's checks. It is NOT a replacement for an obstacle planner.
"""
import torch


def recontact_waypoint(paddle_xy, object_xy, push_direction_xy, mode,
                      contact_distance, object_half=(.02, .02), tool_half=(.015, .02),
                      margin=.006, max_angle_step=.08, position_tolerance=.001,
                      angle_tolerance=.03):
    """Return target_xy, next_mode, approach_ready (not contact-confirmed).

    Modes: 0 outward backoff; 1 bounded shortest arc; 2 radial approach;
    3 reference reached. A new attempt starts at mode 0. contact_distance is
    supplied by the separately verified oriented-rectangle geometry, in metres.
    Directions should be latched for an attempt, not changed using future data.
    Phase-1 use needs an external restriction to the free side of the slot.
    """
    assert margin > 0 and 0 < max_angle_step <= .2
    assert paddle_xy.shape == object_xy.shape == push_direction_xy.shape
    assert paddle_xy.shape[-1] == 2 and mode.shape == paddle_xy.shape[:-1]
    direction = push_direction_xy / push_direction_xy.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    fallback = torch.zeros_like(direction); fallback[..., 0] = 1.
    direction = torch.where((push_direction_xy.norm(dim=-1) > 1e-8)[..., None], direction, fallback)
    offset = paddle_xy - object_xy
    radius = offset.norm(dim=-1)
    radial = torch.where((radius > 1e-8)[..., None], offset / radius[..., None].clamp_min(1e-9), -direction)
    oh = torch.as_tensor(object_half, dtype=paddle_xy.dtype, device=paddle_xy.device)
    th = torch.as_tensor(tool_half, dtype=paddle_xy.dtype, device=paddle_xy.device)
    envelope = oh.norm() + th.norm()
    ring = envelope + margin
    # Even the straight chord between two ring references clears the envelope.
    assert float(ring * torch.cos(torch.as_tensor(max_angle_step / 2))) > float(envelope)
    current_angle = torch.atan2(radial[..., 1], radial[..., 0])
    desired_angle = torch.atan2(-direction[..., 1], -direction[..., 0])
    error = torch.atan2(torch.sin(desired_angle - current_angle), torch.cos(desired_angle - current_angle))
    next_angle = current_angle + error.clamp(-max_angle_step, max_angle_step)
    backoff = object_xy + radial * ring
    orbit = object_xy + torch.stack((next_angle.cos(), next_angle.sin()), -1) * ring
    distance = torch.as_tensor(contact_distance, dtype=paddle_xy.dtype, device=paddle_xy.device).reshape(mode.shape)
    assert bool(torch.all((distance > 0) & (distance < ring)))
    approach = object_xy - direction * distance[..., None]
    next_mode = mode.clone()
    next_mode = torch.where((mode == 0) & (radius >= ring - position_tolerance), 1, next_mode)
    next_mode = torch.where((mode == 1) & (error.abs() <= angle_tolerance)
                            & ((radius - ring).abs() <= position_tolerance), 2, next_mode)
    ready = (mode >= 2) & ((paddle_xy - approach).norm(dim=-1) <= position_tolerance)
    next_mode = torch.where(ready, 3, next_mode)
    target = torch.where((next_mode == 0)[..., None], backoff,
                        torch.where((next_mode == 1)[..., None], orbit, approach))
    return target, next_mode, ready
