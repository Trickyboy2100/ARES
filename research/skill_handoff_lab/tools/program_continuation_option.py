"""Shared temporal interface over unchanged physical feedback primitives.
Reference is frozen throughout a rollout. This is an Option foundation, not a
novel algorithm. No state restoration or original-queue recovery is performed.
"""
import numpy as np

class ContinuationOption:
    def __init__(self, actor, reference, termination='phase'):
        assert termination in ('phase', 'episode', 'chunk', 'insertion_queue')
        self.actor, self.reference, self.termination = actor, reference, termination
        self.active = False
        self.phase = None
        self.events = []

    def __call__(self, s):
        phase = int(round(float(s[49])*7))
        # Grasp loss is observable; reopen control, rather than blindly commit.
        reason = ('grasp_lost' if s[42] < .5 else
                  'phase_end' if self.phase != phase else None)
        if (self.termination == 'chunk'
            or (self.termination == 'phase' and reason)
            or (self.termination == 'insertion_queue' and
                (reason == 'grasp_lost' or (phase == 7 and s[61] <= 0)))):
            self.active = False
        if s[44] < .5:
            action = int(self.actor(s)); boundary = True
        elif self.active:
            action = int(self.reference(s)); boundary = False
        else:
            action = int(self.actor(s)); boundary = True
            self.active = action != 0
        self.phase = phase
        self.events.append(dict(boundary=boundary, action=action, phase=phase,
                                reason=reason, remaining=float(s[41])*100))
        return action

    def transitions(self, transitions):
        """Aggregate actual rewards/durations at policy-over-Option boundaries.
        Undiscounted native finite-horizon return (inherited gamma=1).
        """
        assert len(transitions) == len(self.events)
        out=[]
        for t,e in zip(transitions,self.events):
            if e['boundary']:
                out.append(dict(t, option_termination=self.termination))
            else:
                o=out[-1];o['r']+=t['r'];o['duration']+=t['duration']
                o['next']=t['next'];o['done']=t['done']
        assert sum(t['duration'] for t in out)==100
        return out
