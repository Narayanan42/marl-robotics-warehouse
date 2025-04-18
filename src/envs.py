"""Environment wrappers for MARL training."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Dict, List, Tuple, Any

class MultiAgentEnv:
    def __init__(self, n_agents, obs_dim, action_dim):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_agents, obs_dim), dtype=np.float32)
        self.action_space = spaces.Discrete(action_dim)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

class RWAREWrapper(gym.Wrapper):
    """Wrapper for RWARE environment to conform to MARL interface."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Access the unwrapped environment to get properties
        self.unwrapped_env = env.unwrapped
        self.n_agents = self.unwrapped_env.n_agents
        self.episode_limit = self.unwrapped_env.max_steps  # Fixed attribute name
        # Store the current observation for later use
        self._last_obs = None
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs  # Store observation
        return self._convert_obs(obs), info

    def step(self, actions: List[int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in environment."""
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self._last_obs = obs  # Store observation
        return self._convert_obs(obs), reward, terminated, truncated, info

    def get_obs(self) -> List[np.ndarray]:
        """Get current observations."""
        # Simply return the last stored observation
        if self._last_obs is not None:
            return self._convert_obs(self._last_obs)
        
        # If we don't have observations yet, reset the environment to get them
        obs, _ = self.reset()
        return obs

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        """Get observation for specific agent."""
        obs = self.get_obs()
        return obs[agent_id]

    def get_obs_size(self) -> int:
        """Get size of observation for one agent."""
        # First ensure we have some observations
        if self._last_obs is None:
            self.reset()
            
        # Convert observations and get the first agent's observation shape    
        obs = self._convert_obs(self._last_obs)
        return obs[0].shape[0]

    def get_state(self) -> np.ndarray:
        """Get global state."""
        return np.concatenate(self.get_obs(), axis=0)

    def get_state_size(self) -> int:
        """Get size of global state."""
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self) -> List[List[int]]:
        """Get available actions for all agents."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        """Get available actions for specific agent."""
        if isinstance(self.env.action_space, gym.spaces.Dict):
            n_actions = self.env.action_space[agent_id].n
        elif isinstance(self.env.action_space, gym.spaces.Tuple):
            # For Tuple action spaces, get the space for the specific agent
            n_actions = self.env.action_space[agent_id].n
        elif hasattr(self.env.action_space, 'nvec'):
            # For MultiDiscrete action spaces
            n_actions = self.env.action_space.nvec[agent_id]
        else:
            # Default to Discrete action space
            n_actions = self.env.action_space.n
        return [1] * n_actions

    def get_total_actions(self) -> int:
        """Get total number of actions possible."""
        if isinstance(self.env.action_space, gym.spaces.Dict):
            return self.env.action_space[0].n
        elif isinstance(self.env.action_space, gym.spaces.Tuple):
            # For Tuple action spaces, get the space for the first agent
            return self.env.action_space[0].n
        elif hasattr(self.env.action_space, 'nvec'):
            # For MultiDiscrete action spaces
            return self.env.action_space.nvec[0]
        else:
            # Default to Discrete action space
            return self.env.action_space.n

    def get_stats(self) -> Dict:
        """Get environment statistics."""
        return {}

    def _convert_obs(self, obs: Dict[int, np.ndarray]) -> List[np.ndarray]:
        """Convert dictionary observations to list format."""
        if isinstance(obs, dict):
            return [obs[i] for i in range(self.n_agents)]
        else:
            return obs

    def render(self) -> None:
        """Render environment."""
        return self.env.render()

    def close(self) -> None:
        """Close environment."""
        return self.env.close()

    def seed(self, seed: int = None) -> List[int]:
        """Set environment seed."""
        if seed is None:
            return []
        return [seed]

    def save_replay(self) -> None:
        """Save replay if implemented."""
        pass

    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information."""
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        return env_info
