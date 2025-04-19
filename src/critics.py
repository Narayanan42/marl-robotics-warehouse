import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        
        input_shape = scheme["state"]["vshape"]
        self.output_type = "v"

        # Network architecture
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.n_agents)

    def forward(self, batch_or_states):
        # Handle different input formats
        if isinstance(batch_or_states, dict) or hasattr(batch_or_states, "batch_size"):
            # Input is an episode batch
            states = batch_or_states["state"]  # (batch_size, seq_length, state_dim)
            bs, seq_len = states.size(0), states.size(1)
        else:
            # Input is a state tensor directly
            states = batch_or_states
            # Handle 4D tensor from PPO (batch_size, seq_len, n_agents, state_dim)
            if len(states.shape) == 4:
                # This is likely a tensor with agent dimension - remove or process it
                bs, seq_len = states.shape[0], states.shape[1]
                if states.shape[2] == self.n_agents:
                    # Take the first agent's state representation if it includes agent dimension
                    states = states[:, :, 0, :]
            else:
                # Regular state tensor (batch_size, seq_length, state_dim)
                bs, seq_len = states.shape[0], states.shape[1]
        
        # Reshape for processing through network
        states = states.reshape(-1, states.size(-1))
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        
        # Reshape back to (batch_size, seq_length, n_agents, 1)
        v = v.view(bs, seq_len, -1).unsqueeze(-1)
        return v
    
    def cuda(self):
        """Move the critic to CUDA if available"""
        try:
            if not th.cuda.is_available():
                print("WARNING: CUDA requested but not available for critic! Using CPU.")
                return self
            return super().cuda()
        except RuntimeError as e:
            print(f"WARNING: Error moving critic to CUDA: {e}")
            return self


class ACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ACCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        
        input_shape = scheme["obs"]["vshape"]

        if args.obs_agent_id:
            input_shape += args.n_agents

        if args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        self.output_type = "v"

        # Create separate critic network for each agent
        self.agent_critics = nn.ModuleList([
            AgentCritic(input_shape, args) for _ in range(self.n_agents)
        ])

    def forward(self, batch):
        bs, seq_len = batch.batch_size, batch.max_seq_length
        
        # Collect inputs for all agents
        inputs = []
        inputs.append(batch["obs"])  # (batch_size, seq_len, n_agents, obs_size)
        
        # Add last actions if configured
        if self.args.obs_last_action:
            if seq_len > 1:
                # Create zeros for the first timestep, then use actions for subsequent steps
                last_actions = th.cat([
                    th.zeros_like(batch["actions_onehot"][:, 0:1]),
                    batch["actions_onehot"][:, :-1]
                ], dim=1)
                inputs.append(last_actions)
            else:
                inputs.append(th.zeros_like(batch["actions_onehot"]))
        
        # Add agent IDs if configured
        if self.args.obs_agent_id:
            agent_ids = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0)
            agent_ids = agent_ids.expand(bs, seq_len, -1, -1)
            inputs.append(agent_ids)
        
        # Concatenate all inputs
        inputs = th.cat(inputs, dim=-1)
        
        # Pass inputs through critics for each agent
        vs = []
        for i, critic in enumerate(self.agent_critics):
            agent_input = inputs[:, :, i]  # Get inputs for this agent
            v = critic(agent_input)
            vs.append(v)
        
        # Combine value outputs
        v = th.stack(vs, dim=2)  # Shape: (batch_size, seq_len, n_agents, 1)
        return v
    
    def cuda(self):
        """Move the critic to CUDA if available"""
        try:
            if not th.cuda.is_available():
                print("WARNING: CUDA requested but not available for critic! Using CPU.")
                return self
            return super().cuda()
        except RuntimeError as e:
            print(f"WARNING: Error moving critic to CUDA: {e}")
            return self


class AgentCritic(nn.Module):
    """
    Critic network for a single agent in the ACCritic.
    """
    def __init__(self, input_shape, args):
        super(AgentCritic, self).__init__()
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        
        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        
        v = v.view(orig_shape[:-1] + (1,))
        return v