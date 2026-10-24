"""The phases of the scripted pick-and-place expert (kept here so files that don't need MuJoCo can use them)."""

APPROACH, DESCEND, GRASP, LIFT, CARRY, LOWER, RELEASE, RETREAT = range(8)
PHASE_NAMES = ["approach", "descend", "grasp", "lift", "carry", "lower", "release", "retreat"]
