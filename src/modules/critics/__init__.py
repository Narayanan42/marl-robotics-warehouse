from .centralV import CentralVCritic
from .ac import ACCritic

REGISTRY = {}

REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["ac_critic"] = ACCritic
