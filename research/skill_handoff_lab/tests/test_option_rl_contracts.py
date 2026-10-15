"""Artificial math/unit fixtures only, never a training dataset."""
import unittest
import numpy as np
from research.skill_handoff_lab.cash.qualified_planning import smooth_timing
from research.skill_handoff_lab.cash.contracts import ExecutedTransition,validate_transition_evidence,PLANNING_ATTEMPT_EVIDENCE


class OptionContractTests(unittest.TestCase):
    def test_c2_timing_derivatives_and_endpoints(self):
        q=np.array([[0]*6,[.1]*6,[.07]*6,[.2]*6],float)
        limits={'max_velocity_rad_s':[.1]*6,'max_acceleration_rad_s2':[.2]*6}
        t=smooth_timing(q,limits)
        np.testing.assert_array_equal(t['joints_rad'][0],q[0]);np.testing.assert_array_equal(t['joints_rad'][-1],q[-1])
        self.assertLessEqual(abs(t['velocity_rad_s']).max(),.06+1e-9)
        self.assertLessEqual(abs(t['acceleration_rad_s2']).max(),.12+1e-9)
    def test_budget_failure_is_not_geometric_impossibility(self):
        meta={'budget_before':1,'budget_after':0,'attempt_status':'PLANNER_FAILURE','state_unchanged':True,
              'source_artifacts':[{'role':'ARTIFICIAL_UNIT_FIXTURE'}],'terminal_reason':'BUDGET_EXHAUSTED'}
        t=ExecutedTransition('unit','PREPLACE_PLAN','attempt',None,False,1.,PLANNING_ATTEMPT_EVIDENCE,False,False,(),meta)
        validate_transition_evidence(t,allow_planning=True)
    def test_attempt_cannot_forge_motion_or_success(self):
        t=ExecutedTransition('unit','PREPLACE_PLAN','attempt',None,True,1.,PLANNING_ATTEMPT_EVIDENCE,False,False,(),{})
        with self.assertRaises(ValueError):validate_transition_evidence(t,allow_planning=True)
    def test_budget_is_terminal_not_time_limit_bootstrap(self):
        from research.skill_handoff_lab.cash.env import SkillHandoffEnv
        from research.skill_handoff_lab.cash.contracts import HandoffObservation,HandoffAction
        class Backend:
            def reset(self,seed):
                return HandoffObservation('unit','PREPLACE_PLAN',(0.,),(),(),
                    (HandoffAction('attempt','PREPLACE_PLAN',(0.,),True),),'sha256:unit','sha256:unit')
            def execute(self,observation,action_id):
                return ExecutedTransition('unit','PREPLACE_PLAN','attempt',None,False,1.,PLANNING_ATTEMPT_EVIDENCE,
                    False,False,(),{'budget_before':1,'budget_after':0,'attempt_status':'PLANNER_FAILURE',
                    'state_unchanged':True,'source_artifacts':[{'role':'ARTIFICIAL_UNIT_FIXTURE'}],
                    'terminal_reason':'BUDGET_EXHAUSTED'})
        env=SkillHandoffEnv(Backend(),allow_planning=True);env.reset()
        _,_,terminated,truncated,info=env.step(0)
        self.assertTrue(terminated);self.assertFalse(truncated)
        self.assertEqual(info['transition'].metadata['terminal_reason'],'BUDGET_EXHAUSTED')


if __name__=='__main__':unittest.main()
