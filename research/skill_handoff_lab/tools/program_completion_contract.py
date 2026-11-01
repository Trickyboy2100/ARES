"""Shared observable completion gate; deterministic baseline, no RL novelty.

No native success, hidden goal, bias label or future state is used. Dwell counts
unique physical steps only, with outer hysteresis to avoid one-frame chatter.
"""
from dataclasses import dataclass, asdict
import math

@dataclass
class CompletionGate:
    distance_enter_m: float = .004
    distance_exit_m: float = .0046
    angle_enter_rad: float = .12
    angle_exit_rad: float = .16
    linear_speed_m_s: float = .02
    angular_speed_rad_s: float = .3
    dwell_steps: int = 3
    last_step: int = -1
    count: int = 0
    active: bool = False
    reached: bool = False

    def update(self, step, distance, angle, speed, angular_speed):
        if step < self.last_step:
            raise ValueError("physical step moved backwards; reset gate per episode")
        if not all(math.isfinite(x) and x>=0 for x in [distance,angle,speed,angular_speed]):
            self.active=False;self.count=0;self.last_step=step
            return False
        if step == self.last_step or self.reached:
            return self.reached
        self.last_step = step
        if distance > self.distance_exit_m or angle > self.angle_exit_rad:
            self.active = False
            self.count = 0
        if distance <= self.distance_enter_m and angle <= self.angle_enter_rad:
            self.active = True
        stable = speed <= self.linear_speed_m_s and angular_speed <= self.angular_speed_rad_s
        self.count = self.count + 1 if self.active and stable else 0
        self.reached = self.count >= self.dwell_steps
        return self.reached

    def snapshot(self):
        return asdict(self)
