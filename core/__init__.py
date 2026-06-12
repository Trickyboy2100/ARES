# simforge.core — shared simulation modules
#
# Modules
# -------
# kinematics     FK/IK, URDF chain parsing, pad midpoint (canonical: kinematics.py)
# kinematics_probe  ← same file, kept for backward-compatible imports
# planning       cuRobo-backed path planning, constrained IK paths
# ik_sanity      joint limit checks
# gripper        EG2-4C2 FK, physics drive API, contact box setup
# scene_utils    USD bbox, xform helpers, physics materials, carrier, FixedJoint
