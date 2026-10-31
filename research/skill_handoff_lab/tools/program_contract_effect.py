"""Falsifiable finite-window contact-contract belief features, not a novelty claim.

The same four 21-dimensional observations are exposed to ordinary and effect SAC. The latter
also receives a derived geometric belief (no privileged bias/state labels).
"""
from collections import deque
import numpy as np

class ContractEffect:
    def __init__(self):
        self.history=deque(maxlen=4)
        grid=np.linspace(-.0015,.0015,7)
        self.hypotheses=np.array([(y,z) for y in grid for z in grid])
    def reset(self,obs):
        self.history.clear()
        for _ in range(4):self.history.append(np.array(obs,copy=True))
        return self.features()
    def update(self,obs):self.history.append(np.array(obs,copy=True));return self.features()
    def features(self):
        logp=np.zeros(len(self.hypotheses));used=0
        # Socket clearance is .5mm per side. Orientation can invalidate the
        # translational contact model, hence uncertain observations are censored.
        for obs in self.history:
            x=obs[0]*.025;yz=obs[1:3]*.025;angle=np.linalg.norm(obs[3:6])*.2
            force=np.linalg.norm(obs[9:12])*10;dx=obs[15]*.004;advance=.002+.004*obs[14]
            if angle>.03 or advance<=0 or obs[19]<.5 or obs[20]>.5 or not (-.019<x<-.004):continue
            outside=np.maximum(np.abs(self.hypotheses-yz)-.0005,0)
            feasible=np.exp(-np.sum((outside/.00035)**2,axis=1))
            if dx>.0004 and x>-.014:
                logp+=np.log(.15+.85*feasible);used+=1
            elif force>1. and abs(dx)<.0002:
                logp+=np.log(.3+.7*(1-feasible));used+=1
        p=np.exp(logp-logp.max());p/=p.sum();mean=p@self.hypotheses;var=p@((self.hypotheses-mean)**2)
        entropy=-np.sum(p*np.log(p+1e-12))/np.log(len(p))
        self.posterior=p
        return np.r_[mean/.0015,np.sqrt(var)/.0015,entropy,used/4].astype(np.float32)
    def observation(self,enabled,prior_only=False):
        features=self.features() if enabled else np.zeros(6,dtype=np.float32)
        if enabled and prior_only:
            features=np.r_[np.zeros(2),np.std(self.hypotheses,axis=0)/.0015,1.,0.]
        return np.r_[np.concatenate(self.history),features].astype(np.float32)
