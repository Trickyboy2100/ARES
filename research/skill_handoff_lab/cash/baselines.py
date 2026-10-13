"""Fair baselines over identical candidate sets and observations."""
from __future__ import annotations
import numpy as np

def heuristic_beam_action(action_features,mask,score_index=-1):
    values=np.asarray(action_features)[:,score_index].copy(); values[~np.asarray(mask,bool)]=np.inf
    return int(np.argmin(values))

class SupervisedCandidateRanker:
    """Outcome-trained ranker; labels must come from backend execution records."""
    def __init__(self,random_state=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model=HistGradientBoostingClassifier(max_depth=3,max_iter=80,random_state=random_state)
    def fit(self,x,y,evidence_levels):
        allowed={"CUROBO_ISAAC_EXECUTED","REAL_ROBOT_EXECUTED"}
        if not set(evidence_levels)<=allowed:
            raise ValueError("ranker refuses proxy/hand-authored success labels")
        self.model.fit(np.asarray(x),np.asarray(y)); return self
    def scores(self,x): return self.model.predict_proba(np.asarray(x))[:,1]

def maskable_ppo_status():
    try:
        import gymnasium,stable_baselines3,sb3_contrib
        return {"available":True,"implementation":"sb3_contrib.MaskablePPO",
                "versions":{"gymnasium":gymnasium.__version__,
                  "stable_baselines3":stable_baselines3.__version__,
                  "sb3_contrib":getattr(sb3_contrib,"__version__","UNKNOWN")}}
    except Exception as exc:
        return {"available":False,"implementation":"sb3_contrib.MaskablePPO",
                "reason":type(exc).__name__+":"+str(exc)}
