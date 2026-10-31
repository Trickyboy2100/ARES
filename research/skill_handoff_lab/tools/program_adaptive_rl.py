"""Bounded full-task residual SAC pilot; not a mature RL baseline claim.

SAC acts at the legal three-dimensional skill interface. Every reset physically
executes the prefix; every saved episode reaches the original 200-step horizon.
"""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import numpy as np


class SkillEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, out, seeds, rng_seed=31031, density=1., friction=None,
                 evaluation=False, episode_factory=None, method="sac", prior_only=False):
        super().__init__()
        if episode_factory is None:
            from program_adaptive_contact import Episode
            episode_factory = Episode
        self.episode = episode_factory(density, friction)
        from program_contract_effect import ContractEffect
        self.effect = ContractEffect()
        self.method = method
        self.prior_only = prior_only
        self.out = Path(out)
        self.seeds = list(seeds)
        self.rng = np.random.default_rng(rng_seed)
        self.evaluation = evaluation
        self.action_space = gym.spaces.Box(-1., 1., (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10., 10., (90,), dtype=np.float32)
        self.index = 0
        self.active = False
        self.rows = []
        self.policy_decisions = 0
        self.tail_hold_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.active:
            self.finish_tail("reset_before_terminal")
        options = options or {}
        native_seed = int(options.get("native_seed", self.seeds[self.index % len(self.seeds)]))
        bias = options.get("bias_mm", (float(self.rng.choice([-1.2, 0., 1.2])), 0.))
        self.index += 1
        self.active = True
        obs = self.episode.reset(native_seed, bias)
        self.effect.reset(obs)
        return self.effect.observation(self.method in {"effect_sac", "effect_ln_sac"}, self.prior_only), {"native_seed": native_seed, "bias_mm": list(bias)}

    def _save(self, reason):
        label = f'{"eval" if self.evaluation else "train"}_{self.index:05d}_{reason}'
        row = self.episode.save(self.out / "episodes", label)
        row["completion_reason"] = reason
        row["truncated_training_tail"] = reason != "native_terminal"
        self.rows.append(row)
        self.active = False
        (self.out / "EPISODES.json").write_text(json.dumps(self.rows, indent=2))
        print(json.dumps(dict(event="episode", **row)), flush=True)

    def step(self, action):
        if not self.active:
            raise RuntimeError("reset required before step")
        obs, reward, done = self.episode.step(action)
        self.effect.update(obs)
        obs = self.effect.observation(self.method in {"effect_sac", "effect_ln_sac"}, self.prior_only)
        self.policy_decisions += 1
        info = {}
        if done:
            self._save("native_terminal")
            info = {"is_success": self.rows[-1]["success"]}
        return obs, float(reward), bool(done), False, info

    def finish_tail(self, reason="truncated_training_tail"):
        if not self.active:
            return
        # Training stopping mid-episode must not erase the remaining physics.
        from pull_tool_physics import serial
        ep = self.episode
        start = len(ep.env.records)
        q = np.asarray(serial(ep.env.unwrapped.agent.robot.get_qpos()[0, :7]))
        ep.env.phase = "TRAINING_TAIL_HOLD"
        while len(ep.env.records) < 200:
            ep.env.step(np.r_[q, -1.])
        ep.done = True
        ep.planner.close()
        self.tail_hold_steps += len(ep.env.records) - start
        self._save(reason)

    def counters(self):
        return dict(complete_episodes=len(self.rows),
                    physics_steps=sum(r["steps"] for r in self.rows),
                    prefix_steps=sum(r["prefix_steps"] for r in self.rows),
                    policy_decisions=self.policy_decisions,
                    tail_hold_steps=self.tail_hold_steps,
                    tail_episodes=sum(r["truncated_training_tail"] for r in self.rows),
                    success=sum(r["success"] for r in self.rows),
                    native_terminal_episodes=sum(not r["truncated_training_tail"] for r in self.rows),
                    native_terminal_success=sum(r["success"] for r in self.rows
                                                if not r["truncated_training_tail"]))

    def close(self):
        try:
            self.finish_tail()
        finally:
            self.episode.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sb3-root", type=Path)
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--method", choices=["sac", "effect_sac", "ln_sac", "effect_ln_sac"], default="sac")
    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--model", type=Path)
    p.add_argument("--zero-z", action="store_true", help="Single-channel physical ablation: retain learned y/advance, restore nominal z")
    p.add_argument("--project-y-only", action="store_true", help="Post-training diagnostic: preserve learned lateral-y action, retain nominal z and advance")
    p.add_argument("--prior-only", action="store_true", help="Evaluation ablation: same effect checkpoint, geometric prior but no evidence update")
    p.add_argument("--seed", type=int, default=31031)
    p.add_argument("--seeds", help="Comma-separated development or evaluation scene IDs")
    p.add_argument("--alpha", type=float, default=.02, help="Fixed entropy coefficient (matched pilot default; historical first LN run used .1)")
    p.add_argument("--density", type=float, default=1.)
    p.add_argument("--friction", type=float)
    a = p.parse_args()
    if a.sb3_root:
        import sys
        sys.path.insert(0, str(a.sb3_root))
    if a.steps <= 0 or a.alpha <= 0:
        p.error("steps and alpha must be positive")
    if a.mode == "eval" and a.model is None:
        p.error("eval requires --model")
    # Lazy import permits CPU inspection without constructing the GPU task.
    from stable_baselines3 import SAC
    import stable_baselines3
    import torch
    a.out.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds else
             list(range(728000, 728032) if a.mode == "train" else range(729000, 729008)))
    sources = {}
    for name in ["program_adaptive_rl.py", "program_adaptive_contact.py",
                 "program_adaptive_task.py", "program_charger.py", "pull_tool_physics.py", "program_contract_effect.py", "program_sac_policy.py"]:
        src = Path(__file__).parent / name
        sources[name] = hashlib.sha256(src.read_bytes()).hexdigest()
        (a.out / name).write_bytes(src.read_bytes())
    manifest = dict(method=a.method, mode=a.mode, claim="BOUNDED_PILOT_NOT_MATURE_BASELINE",
                    seeds=seeds, biases_y_mm=[-1.2, 0., 1.2], bias_z_mm=0.,
                    native_horizon=200, native_success_unchanged=True,
                    custom_observation_distribution=True, steps_requested=a.steps,
                    observation_contract="same four raw state/action/effect observations (84); six zeros for SAC or six derived finite-window contract-belief features for effect SAC; same 90-input network and action limits",
                    candidate_claim="UNVERIFIED_GEOMETRIC_EFFECT_FEATURE_MECHANISM_NOT_ESTABLISHED_NOVEL",
                    sb3_version=stable_baselines3.__version__, sources=sources,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    (a.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    env = SkillEnv(a.out, seeds, a.seed, a.density, a.friction, a.mode == "eval", method=a.method, prior_only=a.prior_only)
    model = None
    started = time.monotonic()
    error = None
    try:
        if a.mode == "train":
            from program_sac_policy import CriticLayerNormPolicy
            policy = CriticLayerNormPolicy if a.method in {"ln_sac", "effect_ln_sac"} else "MlpPolicy"
            model = SAC(policy, env, policy_kwargs=dict(net_arch=[64, 64]),
                        learning_rate=3e-4, buffer_size=max(10000, a.steps + 1),
                        learning_starts=128, batch_size=64, train_freq=1,
                        gradient_steps=1, ent_coef=a.alpha, gamma=.99, tau=.005,
                        seed=a.seed, device="cpu", verbose=1)
            model.learn(total_timesteps=a.steps, log_interval=1, progress_bar=False)
            env.finish_tail()
            model.save(a.out / "MODEL.zip")
            model.save_replay_buffer(a.out / "REPLAY_BUFFER.pkl")
        else:
            model = SAC.load(a.model, device="cpu")
            for seed in seeds:
                for bias in [-1.2, 0., 1.2]:
                    obs, _ = env.reset(options={"native_seed": seed, "bias_mm": [bias, 0.]})
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        if a.zero_z:
                            action = np.array([action[0], 0., action[2]], dtype=np.float32)
                        if a.project_y_only:
                            action = np.array([action[0], 0., 0.], dtype=np.float32)
                        obs, _, done, _, _ = env.step(action)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        try:
            env.close()
        finally:
            updates = int(getattr(model, "_n_updates", 0)) if a.mode == "train" else 0
            result = dict(**env.counters(), optimizer_updates=updates,
                          actor_optimizer_steps=updates, critic_optimizer_steps=updates,
                          entropy_optimizer_steps=0, wall_s=time.monotonic() - started,
                          error=error)
            (a.out / "RESULTS.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(dict(event="finished", **result)), flush=True)


if __name__ == "__main__":
    main()
