"""Policy network implementations for agents."""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transforms import OneHot


class BasicMAC:
    """Basic multi-agent controller that shares parameters between agents."""
    
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type
        
        # Define the action selector based on args
        self.action_selector = self._get_action_selector(args)
        
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """Select actions for the agents."""
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        """Forward pass through the agent network."""
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimize their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = F.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        """Initialize hidden states for RNN type networks."""
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # (batch, n_agents, hidden_size)

    def parameters(self):
        """Return parameters of the agent network."""
        return self.agent.parameters()

    def load_state(self, other_mac):
        """Load network state from another MAC instance."""
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        """Move the agent to CUDA."""
        self.agent.cuda()

    def save_models(self, path):
        """Save agent models to disk."""
        th.save(self.agent.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        """Load agent models from disk."""
        self.agent.load_state_dict(th.load(f"{path}/agent.th", map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        """Create the agent network."""
        self.agent = RNNAgent(input_shape, self.args)

    def _build_inputs(self, batch, t):
        """Create inputs for the agent network."""
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # (bs, n_agents, obs_dim)
        
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
                
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        """Calculate the input shape for the agent network."""
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
    
    def _get_action_selector(self, args):
        """Create an action selector based on args."""
        if args.action_selector == "epsilon_greedy":
            return EpsilonGreedyActionSelector(args)
        elif args.action_selector == "multinomial":
            return MultinomialActionSelector(args)
        elif args.action_selector == "soft_policies":
            return SoftPoliciesSelector(args)
        else:
            raise ValueError(f"Unsupported action selector: {args.action_selector}")


class RNNAgent(nn.Module):
    """RNN-based agent network."""
    
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        """Initialize hidden states."""
        # Make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """Forward pass through the network."""
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h


class DecayThenFlatSchedule:
    """Epsilon decay schedule for exploration."""
    
    def __init__(self, start, finish, time_length, decay="exp"):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, t):
        """Evaluate epsilon at time t."""
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * t)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- t / self.exp_scaling)))
        else:
            raise ValueError(f"Unknown decay type: {self.decay}")


class MultinomialActionSelector:
    """Selects actions using a multinomial distribution over agent outputs."""
    
    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                             decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """Select actions using multinomial distribution with epsilon exploration."""
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=-1)[1]
        else:
            picked_actions = th.distributions.Categorical(masked_policies).sample().long()

        return picked_actions


class EpsilonGreedyActionSelector:
    """Select actions using epsilon-greedy."""
    
    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                             decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """Select actions using epsilon-greedy with action masking."""
        # Assuming agent_inputs is a batch of Q-Values for each agent (batch, n_agents, n_actions)
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # Mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # Should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = th.distributions.Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=-1)[1]
        return picked_actions


class SoftPoliciesSelector:
    """Selects actions by directly sampling from policy outputs."""
    
    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """Sample actions from policy distribution."""
        # Mask before sampling to ensure valid actions
        masked_probs = agent_inputs.clone()
        masked_probs[avail_actions == 0] = 0.0
        
        # Normalize if needed
        if masked_probs.sum(dim=-1, keepdim=True).min() <= 0:
            # Some agents have no valid actions, add a small constant to avoid division by zero
            masked_probs = masked_probs + 1e-8
            
        # Normalize
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
        
        # Sample from the distribution
        m = th.distributions.Categorical(masked_probs)
        picked_actions = m.sample().long()
        
        return picked_actions