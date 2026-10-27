"""Tensor geometry for the synthetic contact pilot; no Isaac dependency."""
import torch


def to_task_xy(x, forward):
    c, s = forward[:, 0], forward[:, 1]
    return torch.stack((c*x[:, 0]+s*x[:, 1], -s*x[:, 0]+c*x[:, 1]), -1)


def from_task_xy(x, forward):
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), -1)
    return forward*x[:, :1]+lateral*x[:, 1:2]


def rear_alignment(paddle, obj, command, forward):
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), -1)
    relative = to_task_xy(paddle-obj, forward)
    eligible = (relative[:, 0] < -.037) & (relative[:, 1].abs() > .003)
    delta = lateral*((obj-command)*lateral).sum(-1, keepdim=True).clamp(-.0012, .0012)
    return eligible, delta


def task_observation(old, extra, forward):
    """v6 33+23 layout; preserve phase/time/dwell and physical scalar context."""
    old=old.clone();extra=extra.clone()
    for lo in [12,15,18,21,24,27]:old[:,lo:lo+2]=to_task_xy(old[:,lo:lo+3],forward)
    for lo in [4,7,10]:extra[:,lo:lo+2]=to_task_xy(extra[:,lo:lo+3],forward)
    phi=torch.atan2(forward[:,1],forward[:,0]);h=-phi/2;c,s=h.cos(),h.sin()
    w,x,y,z=extra[:,:4].unbind(-1)
    extra[:,:4]=torch.stack((c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w),-1)
    # geometry begins13: gap,goal_x,slot_y,mass,mu,cos(phi),sin(phi).
    # Index20 is dwell and must never be overwritten.
    extra[:,18]=1.;extra[:,19]=0.
    return old,extra
