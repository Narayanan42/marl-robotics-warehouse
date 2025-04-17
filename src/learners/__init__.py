from .actor_critic_learner import ActorCriticLearner
from .ppo_learner import PPOLearner


REGISTRY = {}
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["ppo_learner"] = PPOLearner
