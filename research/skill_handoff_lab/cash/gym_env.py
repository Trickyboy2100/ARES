"""Fixed-size Gymnasium adapter for variable legal handoff candidates."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .env import SkillHandoffEnv
from .contracts import STAGES

class GymSkillHandoffEnv(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self,backend,state_dim,action_dim,max_actions,allow_planning=True):
        super().__init__()
        self.core=SkillHandoffEnv(backend,allow_planning=allow_planning)
        self.state_dim=state_dim; self.action_dim=action_dim; self.max_actions=max_actions
        self.action_space=spaces.Discrete(max_actions)
        self.observation_space=spaces.Dict({
            "state":spaces.Box(-np.inf,np.inf,(state_dim,),np.float32),
            "action_features":spaces.Box(-np.inf,np.inf,(max_actions,action_dim),np.float32),
            "stage":spaces.Box(0,1,(len(STAGES),),np.float32)})
        self._mask=np.zeros(max_actions,dtype=bool)
    def _pack(self,obs):
        result={"state":np.zeros(self.state_dim,np.float32),
                "action_features":np.zeros((self.max_actions,self.action_dim),np.float32),
                "stage":np.zeros(len(STAGES),np.float32)}
        self._mask[:]=False
        if obs["stage"]=="TERMINAL": return result
        if len(obs["state"])!=self.state_dim: raise ValueError("state dimension mismatch")
        features=obs["action_features"]; n=len(features)
        if n>self.max_actions or features.shape!=(n,self.action_dim):
            raise ValueError("candidate dimensions exceed declared space")
        result["state"][:]=obs["state"]; result["action_features"][:n]=features
        result["stage"][STAGES.index(obs["stage"])]=1
        self._mask[:n]=obs["action_mask"]
        if not self._mask.any(): raise ValueError("nonterminal state has no legal action")
        return result
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        obs,info=self.core.reset(seed=int(self.np_random.integers(2**31)),options=options)
        return self._pack(obs),info
    def action_masks(self): return self._mask.copy()
    def step(self,action):
        obs,reward,terminated,truncated,info=self.core.step(int(action))
        return self._pack(obs),reward,terminated,truncated,info
