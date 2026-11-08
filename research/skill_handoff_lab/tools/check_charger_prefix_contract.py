"""Regression: failed predecessor must not become an executable successor."""
from program_native_charger_pace import prefix_failure
assert prefix_failure(dict(prefix_steps=67,error='CHARGER_NOT_GRASPED'))=='CHARGER_NOT_GRASPED'
assert prefix_failure(dict(prefix_steps=88,error='CHARGER_LOST_ON_LIFT'))=='CHARGER_LOST_ON_LIFT'
assert prefix_failure(dict(prefix_steps=90,error='PLAN_ALIGN'))=='PLAN_ALIGN'
assert prefix_failure(dict(prefix_steps=121,error='SERVO_IK_FAILED')) is None
assert prefix_failure(dict(prefix_steps=121,error=None)) is None
assert prefix_failure(dict(prefix_steps=200,error='HORIZON'))=='HORIZON'
print('PASS: preserve prefix failure, never leak a later servo failure into replay start')
