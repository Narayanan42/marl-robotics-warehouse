"""
Policy optimization algorithms for MARL agents.
"""
import copy
import torch as th
from torch.optim import Adam

from utils import RunningMeanStd
from buffers import EpisodeBatch
from critics import CentralVCritic, ACCritic

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCriticLearner:
    """Actor-critic learner for MARL with centralized value function."""
    def __init__(self, mac, scheme, args):
        self.args = args
        self.mac = mac
        self.n_agents = args.n_agents
        
        # Parameters and optimizer
        self.params = list(mac.parameters())
        
        # Initialize critic
        critic_type = getattr(args, "critic_type", "central_v")
        if critic_type == "cv_critic" or critic_type == "central_v":
            self.critic = CentralVCritic(scheme, args)
        else:
            self.critic = ACCritic(scheme, args)
            
        self.target_critic = copy.deepcopy(self.critic)
        
        self.params += list(self.critic.parameters())
        
        # Create optimizer
        self.optimizer = Adam(params=self.params, lr=args.lr)
        
        # Target update parameters
        self.last_target_update_episode = 0
        self.target_update_interval = getattr(args, "target_update_interval", getattr(args, "target_update_interval_or_tau", 200))
        
        # Stats and tracking
        self.device = args.device
        self.train_info = {}
        
        # Set up normalization if needed
        if getattr(args, "standardise_returns", False):
            self.ret_ms = RunningMeanStd(shape=(1,), device=self.device)
        if getattr(args, "standardise_rewards", False):
            self.rew_ms = RunningMeanStd(shape=(1,), device=self.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # [bs, t, n_agents, n_actions]
        
        # Pick the Q-values for the actions taken
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        
        # Get critic values
        critic_inputs = batch["state"]
        critic_mask = mask.repeat(1, 1, self.n_agents)
        
        q_vals = self.critic(batch)
        q_vals = q_vals[:, :-1]  # Drop last timestep
        
        # Calculate targets
        with th.no_grad():
            target_q_vals = self.target_critic(batch)[:, 1:]
            targets = rewards + self.args.gamma * (1 - terminated) * target_q_vals
            
        # Standardize if needed
        if getattr(self.args, "standardise_returns", False):
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)
        
        # Calculate TD-error
        td_error = (targets - q_vals)
        masked_td_error = td_error * mask.unsqueeze(-1)
        
        # Critic loss
        critic_loss = (masked_td_error ** 2).sum() / mask.sum()
        
        # Actor loss
        actions_onehot = batch["actions_onehot"][:, :-1]
        pi = mac_out[:, :-1]
        
        # Advantage calculation
        advantage = (targets - q_vals).detach()
        
        # Policy gradient loss with entropy regularization
        pi_taken = th.sum(pi * actions_onehot, dim=-1)
        log_pi_taken = th.log(pi_taken + 1e-10)
        
        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
        actor_loss = -(log_pi_taken * advantage.squeeze(-1)).sum() / mask.sum()
        entropy_loss = entropy.sum() / mask.sum()
        
        # Total loss with entropy regularization
        loss = actor_loss - self.args.entropy_coef * entropy_loss + self.args.critic_coef * critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimizer.step()
        
        # Update target network if needed
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
            
        # Save training info
        self.train_info = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "entropy": entropy_loss.item(),
            "grad_norm": grad_norm.item(),
            "advantage_mean": advantage.mean().item(),
            "pi_max": pi.max(dim=-1)[0].mean().item(),
            "train_step": t_env
        }
    
    def _update_targets(self):
        """Update target network parameters."""
        tau = getattr(self.args, "target_update_interval_or_tau", 0.01)
        if tau < 1:  # Soft update
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        else:  # Hard update
            self.target_critic.load_state_dict(self.critic.state_dict())

    def cuda(self):
        """Transfer models to GPU."""
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        """Save models to disk."""
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), f"{path}/critic.th")
        th.save(self.optimizer.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        """Load models from disk."""
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load(f"{path}/critic.th", map_location=lambda storage, loc: storage))
        self.optimizer.load_state_dict(th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage))
        
    def get_train_info(self):
        """Return training metrics collected during last update."""
        return self.train_info

class PPOLearner:
    """PPO learner implementation for MARL with MAC structure."""
    def __init__(self, mac, scheme, args):
        self.args = args
        self.mac = mac
        # Remove logger reference
        self.logger = None

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0
        self.device = args.device
        
        # Initialize tracking variables
        self.train_info = {}
        self.log_stats_t = 0

        # Create critics based on type
        critic_type = getattr(args, "critic_type", "central_v")
        if critic_type == "cv_critic" or critic_type == "central_v":
            self.critic = CentralVCritic(scheme, args)
        else:
            self.critic = ACCritic(scheme, args)
            
        self.target_critic = copy.deepcopy(self.critic)
        
        # Add critic parameters to optimizer
        self.params += list(self.critic.parameters())
        
        # Configure optimizer
        self.optimizer = optim.Adam(self.params, lr=args.lr)
        
        # Initialize PPO parameters
        self.clip_param = getattr(args, "eps_clip", 0.2)
        self.ppo_epochs = getattr(args, "epochs", 4)
        
        # Get the target update interval (handle both attribute names)
        self.target_update_interval = getattr(args, "target_update_interval", getattr(args, "target_update_interval_or_tau", 200))
        
        # Set up value stats for normalization if needed
        if getattr(self.args, "standardise_returns", False):
            self.ret_ms = RunningMeanStd(shape=(1,), device=self.device)
        if getattr(self.args, "standardise_rewards", False):
            self.rew_ms = RunningMeanStd(shape=(1,), device=self.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get relevant quantities from batch
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # [bs, t, n_agents, n_actions]

        # Get critic values
        critic_inputs = self._build_critic_inputs(batch)
        q_vals = self.critic(critic_inputs)

        # Calculate targets and advantages
        with th.no_grad():
            target_q_vals = self.target_critic(critic_inputs)
            
            if rewards.dim() != target_q_vals.dim() or rewards.shape[-1] != target_q_vals.shape[-1]:
                # Check if rewards need to be expanded for per-agent values
                if target_q_vals.dim() == 4 and rewards.dim() == 3:
                    rewards = rewards.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1)
                    # Also expand terminated tensor
                    terminated = terminated.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1)
            
            # Now calculate targets with properly shaped tensors
            targets = rewards + self.args.gamma * (1 - terminated) * target_q_vals[:, 1:]

        if getattr(self.args, "standardise_returns", False):
            with th.no_grad():
                targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        with th.no_grad():
            advantages = (targets - q_vals[:, :-1]).detach()
        
        # For actor loss, we need to make sure advantages are correctly shaped
        # If advantages are 4D [bs, t, n_agents, 1] but actor outputs are 3D [bs, t, n_agents]
        if advantages.dim() == 4 and advantages.shape[-1] == 1:
            # Remove the last singleton dimension
            advantages_for_pg = advantages.squeeze(-1)
        else:
            advantages_for_pg = advantages
        
        with th.no_grad():
            actions_onehot = batch["actions_onehot"][:, :-1]
            old_log_probs = ((mac_out[:, :-1].detach() + 1e-10).log() * actions_onehot).sum(dim=-1)

        # PPO update loop
        for epoch in range(self.ppo_epochs):
            # Calculate new action probabilities
            self.mac.init_hidden(batch.batch_size)
            new_mac_out = []
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                new_mac_out.append(agent_outs)
            new_mac_out = th.stack(new_mac_out, dim=1)  # [bs, t, n_agents, n_actions]
            
            # Log probabilities with new policy
            new_log_probs = ((new_mac_out + 1e-10).log() * actions_onehot).sum(dim=-1)

            # Calculate ratios and PPO loss
            ratios = th.exp(new_log_probs - old_log_probs)
            
            if ratios.shape != advantages_for_pg.shape:
                if ratios.dim() == advantages_for_pg.dim() - 1 and advantages_for_pg.shape[-1] == 1:
                    advantages_for_pg = advantages_for_pg.squeeze(-1)
                elif ratios.dim() == advantages_for_pg.dim():
                    pass
                else:
                    try:
                        advantages_for_pg = advantages_for_pg.reshape(ratios.shape)
                    except:
                        if advantages_for_pg.shape[2] != ratios.shape[2]:
                            advantages_for_pg = advantages_for_pg.mean(dim=2, keepdim=True)
                            if advantages_for_pg.shape[-1] == 1 and advantages_for_pg.dim() > ratios.dim():
                                advantages_for_pg = advantages_for_pg.squeeze(-1)
                            
            surr1 = ratios * advantages_for_pg
            surr2 = th.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_for_pg
            actor_loss = -th.min(surr1, surr2).mean()
            if epoch > 0:  
                q_vals = self.critic(critic_inputs)

            critic_mask = mask
            if q_vals.dim() > mask.dim():

                for _ in range(q_vals.dim() - mask.dim()):
                    critic_mask = critic_mask.unsqueeze(-1)

                if q_vals.shape[2] > 1 and critic_mask.shape[2] == 1:
                    critic_mask = critic_mask.expand(-1, -1, q_vals.shape[2], -1)
            

            critic_loss = ((q_vals[:, :-1] - targets.detach()) * critic_mask).pow(2).sum() / (critic_mask.sum() + 1e-8)
            
            # Combined loss
            loss = actor_loss + self.args.critic_coef * critic_loss

            # Update parameters
            self.optimizer.zero_grad()
            try:
                if epoch < self.ppo_epochs - 1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
            except RuntimeError as e:
                print(f"Backward error in epoch {epoch}: {e}")
                loss.backward()
                
            # Clip gradients if needed
            if getattr(self.args, "grad_norm_clip", None) is not None:
                th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimizer.step()

        # Update target networks if needed
        if (episode_num - self.last_target_update_episode) / self.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # Store training metrics instead of logging them
        self.train_info = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "advantage_mean": advantages.mean().item(),
            "ratio_mean": ratios.mean().item(),
            "train_step": t_env
        }

    def _build_critic_inputs(self, batch):
        """Build inputs for centralized critic."""
        bs = batch.batch_size
        max_t = batch.max_seq_length
        
        # Get state information
        states = batch["state"]  # Shape: (batch_size, seq_length, state_dim)
        
        # For centralized critic, we can return the state directly
        if critic_type := getattr(self.args, "critic_type", "central_v"):
            if critic_type in ["central_v", "cv_critic"]:
                return states
        if self.args.obs_agent_id:
            agent_ids = th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
            # Expand state to include the agent dimension
            expanded_state = states.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1)
            # Concatenate state with agent IDs
            return th.cat([expanded_state, agent_ids], dim=-1)
        
        return states

    def _update_targets(self):
        """Update target network parameters."""
        tau = getattr(self.args, "target_update_interval_or_tau", 0.01)
        if tau < 1:  # Soft update
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        else:  # Hard update
            self.target_critic.load_state_dict(self.critic.state_dict())

    def cuda(self):
        """Transfer models to GPU."""
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        """Save models to disk."""
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), f"{path}/critic.th")
        th.save(self.optimizer.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        """Load models from disk."""
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load(f"{path}/critic.th", map_location=lambda storage, loc: storage))
        self.optimizer.load_state_dict(th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage))
        
    def get_train_info(self):
        """Return training metrics collected during last update."""
        return self.train_info