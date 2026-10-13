"""Artificial fixtures only: API/evidence checks, never research success data."""
import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from research.skill_handoff_lab.cash.contracts import *
from research.skill_handoff_lab.cash.gym_env import GymSkillHandoffEnv

class FixtureBackend:
    def reset(self,seed):
        self.stage=0
        return self.obs()
    def obs(self):
        stage=STAGES[self.stage]
        return HandoffObservation("fixture",stage,(0.,),(1.,),(2.,),
            (HandoffAction("legal",stage,(0.,),True),HandoffAction("illegal",stage,(1.,),False)),
            "sha256:fixture","sha256:history")
    def execute(self,obs,action):
        self.stage+=1
        return ExecutedTransition("fixture",obs.stage,action,self.obs() if self.stage<3 else None,
            True if self.stage==3 else None,1.,PLANNING_EVIDENCE,True,False,(),
            {"independent_validation":dict.fromkeys(("collision","self_collision","attached_object",
                "soft_limits","orientation","scene_binding"),True),
             "source_artifacts":[{"path":"fixture-only","sha256":"fixture"}]})

class CashGymTests(unittest.TestCase):
    def test_three_stage_api_and_terminal_shape(self):
        class UnmaskedFixture(FixtureBackend):
            def obs(self):
                o=super().obs()
                return HandoffObservation(o.episode_id,o.stage,o.robot_state,o.remaining_goal,
                    o.successor_constraints,(o.actions[0],),o.scene_revision,o.history_digest)
        check_env(GymSkillHandoffEnv(UnmaskedFixture(),3,1,1),skip_render_check=True)
        env=GymSkillHandoffEnv(FixtureBackend(),3,1,4)
        obs,_=env.reset(seed=4)
        for i in range(3):
            self.assertEqual(env.action_masks().tolist(),[True,False,False,False])
            obs,_,done,_,_=env.step(0)
            self.assertEqual(done,i==2); self.assertTrue(env.observation_space.contains(obs))
    def test_timeout_not_negative_label(self):
        value=ExecutedTransition("e",STAGES[0],"a",None,False,0.,PLANNING_EVIDENCE,
            False,False,("TIMEOUT",),{"source_artifacts":["fixture"],"failure_class":"TIMEOUT"})
        with self.assertRaises(ValueError):validate_transition_evidence(value,True)
    def test_maskable_ppo_cpu_integration_fixture_only(self):
        from sb3_contrib import MaskablePPO
        env=GymSkillHandoffEnv(FixtureBackend(),3,1,4)
        model=MaskablePPO("MultiInputPolicy",env,n_steps=8,batch_size=8,n_epochs=1,
            policy_kwargs={"net_arch":[8]},device="cpu",seed=13,verbose=0)
        model.learn(total_timesteps=16)
        obs,_=env.reset(seed=13)
        self.assertEqual(int(model.predict(obs,action_masks=env.action_masks())[0]),0)
    def test_unchecked_collision_not_positive_label(self):
        b=FixtureBackend(); obs=b.reset(0); t=b.execute(obs,"legal")
        t=ExecutedTransition(t.episode_id,t.stage,t.action_id,None,True,0.,PLANNING_EVIDENCE,
            True,False,(),{"source_artifacts":["fixture"],"independent_validation":{}})
        with self.assertRaises(ValueError):validate_transition_evidence(t,True)
if __name__=="__main__":unittest.main()
