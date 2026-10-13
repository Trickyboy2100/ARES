"""Train/evaluate CPU MaskablePPO on qualified immutable planning trees."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from research.skill_handoff_lab.cash.cache_backend import CachedPlanningBackend
from research.skill_handoff_lab.cash.gym_env import GymSkillHandoffEnv

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--steps",type=int,default=4096)
    p.add_argument("--seed",type=int,default=17); p.add_argument("--envs",type=int,default=2)
    p.add_argument("--eval-episodes",type=int,default=100); a=p.parse_args()
    if not 1<=a.envs<=4: raise ValueError("use 1..4 CPU environments")
    data=json.loads(a.dataset.read_text()); dims=data["dimensions"]
    # Both splits and all artifact hashes are checked before any output/training.
    CachedPlanningBackend(a.dataset,"train"); CachedPlanningBackend(a.dataset,"test")
    a.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(2)
    def make(split,index):
        return lambda:Monitor(GymSkillHandoffEnv(CachedPlanningBackend(a.dataset,split),**dims),
            filename=str(a.output/(split+str(index))))
    env=DummyVecEnv([make("train",i) for i in range(a.envs)])
    model=MaskablePPO("MultiInputPolicy",env,device="cpu",seed=a.seed,n_steps=64,
        batch_size=64,n_epochs=4,policy_kwargs={"net_arch":[64,64]},verbose=0)
    model.set_logger(__import__("stable_baselines3.common.logger",fromlist=["configure"]).configure(
        str(a.output),["csv"]))
    started=time.monotonic(); model.learn(total_timesteps=a.steps)
    elapsed=time.monotonic()-started; model.save(a.output/"maskable_ppo")
    results=[]
    for policy in ("random","maskable_ppo"):
        test=GymSkillHandoffEnv(CachedPlanningBackend(a.dataset,"test"),**dims)
        rng=np.random.default_rng(a.seed)
        for index in range(a.eval_episodes):
            obs,_=test.reset(seed=a.seed+index); done=False; total=0
            while not done:
                mask=test.action_masks()
                action=int(rng.choice(np.flatnonzero(mask))) if policy=="random" else int(
                    model.predict(obs,deterministic=True,action_masks=mask)[0])
                obs,reward,done,_,info=test.step(action); total+=reward
            result=info["result"]
            results.append({"policy":policy,"seed":a.seed+index,"episode_id":result.episode_id,
                "success":result.success,"return":total,"cost":result.total_cost,
                "evidence_level":result.evidence_level,"failure_codes":result.failure_codes})
    (a.output/"evaluation.json").write_text(json.dumps(results,indent=2)+"\n")
    manifest={"dataset_sha256":hashlib.sha256(a.dataset.read_bytes()).hexdigest(),"seed":a.seed,
        "steps":model.num_timesteps,"wall_s":elapsed,"steps_per_s":model.num_timesteps/elapsed,
        "evidence":"CACHED_VALIDATED_PLANNING","physical_simulation":False,
        "training_algorithm":"MaskablePPO baseline; CASH auxiliary head not integrated"}
    (a.output/"training_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest))
if __name__=="__main__":main()
