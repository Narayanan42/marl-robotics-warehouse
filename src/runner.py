
from functools import partial
from multiprocessing import Pipe, Process
import pickle
import cloudpickle

import numpy as np
import torch as th

from buffers import EpisodeBatch


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        return pickle.loads(ob)


def env_worker(remote, env_fn):

    if hasattr(env_fn, 'x'):
        try:
            env = env_fn.x() 
        except TypeError:            env = env_fn.x
    else:
        # Direct callable
        env = env_fn()
        
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            _, reward, terminated, truncated, env_info = env.step(actions)
            terminated = terminated or truncated
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send(
                {
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                }
            )
        elif cmd == "reset":
            env.reset()
            remote.send(
                {
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs(),
                }
            )
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "render":
            env.render()
        elif cmd == "save_replay":
            env.save_replay()
        else:
            raise NotImplementedError


class ParallelRunner:
    """Runner that handles parallel execution of environments"""
    def __init__(self, env_fn, args, logger=None):
        self.args = args
        # Make logger optional
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )

        # Create environments with proper arguments
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] = self.args.seed + i
            env_args[i]["common_reward"] = self.args.common_reward
            env_args[i]["reward_scalarisation"] = self.args.reward_scalarisation
        
        try:
            test_env = env_fn(**env_args[0])
            self.ps = [
                Process(
                    target=env_worker,
                    args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
                )
                for env_arg, worker_conn in zip(env_args, self.worker_conns)
            ]
            test_env.close()
        except TypeError:
            # Function doesn't accept arguments, use directly
            self.ps = [
                Process(
                    target=env_worker,
                    args=(worker_conn, CloudpickleWrapper(env_fn)),
                )
                for worker_conn in self.worker_conns
            ]

        # Start processes
        for p in self.ps:
            p.daemon = True
            p.start()

        # Get environment info
        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        # Initialize tracking variables
        self.t = 0
        self.t_env = 0
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        """Set up the batch, schemes, and controller"""
        # Set device safely
        device = self.args.device
        try:
            if device == "cuda" and not th.cuda.is_available():
                print("CUDA requested but not available! Using CPU.")
                device = "cpu"
                self.args.device = device
        except RuntimeError as e:
            print(f"CUDA error: {e}. Using CPU instead.")
            device = "cpu"
            self.args.device = device
            
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=device,
        )
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        self.parent_conns[0].send(("save_replay", None))

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        """Reset environments and create a new batch"""
        self.batch = self.new_batch()

        # Reset all environments
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        # Collect initial state information
        pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        # Update batch with initial data
        self.batch.update(pre_transition_data, ts=0)

        # Reset counters
        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        """Run a complete episode batch across all envs"""
        self.reset()

        all_terminated = False
        if self.args.common_reward:
            episode_returns = [0 for _ in range(self.batch_size)]
        else:
            episode_returns = [
                np.zeros(self.args.n_agents) for _ in range(self.batch_size)
            ]
            
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = []

        while True:
            try:
                actions = self.mac.select_actions(
                    self.batch,
                    t_ep=self.t,
                    t_env=self.t_env,
                    bs=envs_not_terminated,
                    test_mode=test_mode,
                )
                cpu_actions = actions.to("cpu").numpy()
            except RuntimeError as e:
                print(f"Error in select_actions: {e}")
                avail_actions = self.batch["avail_actions"][:, self.t][envs_not_terminated]
                random_actions = np.array([np.random.choice(np.where(aa[i] > 0)[0]) 
                                         for i, aa in enumerate(avail_actions.cpu().numpy())])
                actions = th.tensor(random_actions, device=self.args.device)
                cpu_actions = random_actions

            # Update the actions taken
            actions_chosen = {"actions": actions.unsqueeze(1)}
            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  
                    if not terminated[idx]:  
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1
                    if idx == 0 and test_mode and self.args.render:
                        parent_conn.send(("render", None))

            # Update active environments list
            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            post_transition_data = {"reward": [], "terminated": []}
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    
                    reward = data["reward"]
                    
                    if isinstance(reward, (list, np.ndarray)) and len(np.array(reward).shape) > 0:
                        scalarisation = getattr(self.args, "reward_scalarisation", "sum")
                        if scalarisation == "mean":
                            scalar_reward = np.mean(reward)
                        else:  
                            scalar_reward = np.sum(reward)
                        post_transition_data["reward"].append((scalar_reward,))
                    else:
                        post_transition_data["reward"].append((reward,))

                    if isinstance(reward, (list, np.ndarray)):
                        if self.args.common_reward:
                            scalarisation = getattr(self.args, "reward_scalarisation", "sum")
                            if scalarisation == "mean":
                                episode_returns[idx] += np.mean(reward)
                            else:  # Default to sum
                                episode_returns[idx] += np.sum(reward)
                        else:
                            if len(np.array(reward).shape) > 0 and len(reward) == len(episode_returns[idx]):
                                episode_returns[idx] += np.array(reward)
                            else:
                                try:
                                    episode_returns[idx] += np.array(reward)
                                except ValueError:
                                    episode_returns[idx] += np.sum(reward)
                    else:
                        episode_returns[idx] += reward
                    
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )
            self.t += 1

            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        # Collect stats
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        
        # Update stats from environments
        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos]) if infos
            }
        )
        
        # Episode statistics
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        cur_returns.extend(episode_returns)

        # Calculate statistics instead of logging them
        n_test_runs = max(1, getattr(self.args, "test_nepisode", 20) // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._calculate_stats(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= getattr(self.args, "runner_log_interval", 10000):
            self._calculate_stats(cur_returns, cur_stats, log_prefix)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _calculate_stats(self, returns, stats, prefix):
        """Calculate statistics instead of logging them"""
        # Store stats in dictionaries for later use
        self.return_stats = {}
        
        # Handle different types of returns based on common reward setting
        if getattr(self.args, "common_reward", True):
            self.return_stats[f"{prefix}return_mean"] = np.mean(returns)
            self.return_stats[f"{prefix}return_std"] = np.std(returns)
            
            # Print stats if no logger
            if self.logger is None:
                mean_return = np.mean(returns)
                print(f"{prefix.capitalize()}Return (mean): {mean_return:.3f}")
        else:
            # Calculate per-agent stats
            for i in range(self.args.n_agents):
                mean_return = np.array(returns)[:, i].mean()
                std_return = np.array(returns)[:, i].std()
                self.return_stats[f"{prefix}agent_{i}_return_mean"] = mean_return
                self.return_stats[f"{prefix}agent_{i}_return_std"] = std_return
                
                # Print stats if no logger
                if self.logger is None:
                    print(f"{prefix.capitalize()}Agent {i} return (mean): {mean_return:.3f}, (std): {std_return:.3f}")
                
            total_returns = np.array(returns).sum(axis=-1)
            total_mean = total_returns.mean()
            total_std = total_returns.std()
            self.return_stats[f"{prefix}total_return_mean"] = total_mean
            self.return_stats[f"{prefix}total_return_std"] = total_std
            
            # Print stats if no logger
            if self.logger is None:
                print(f"{prefix.capitalize()}Total return (mean): {total_mean:.3f}, (std): {total_std:.3f}")
            
        # Clear returns after calculating stats
        returns.clear()

        # Calculate other stats
        self.env_stats = {}
        for k, v in stats.items():
            if k != "n_episodes":
                stat_value = v / stats["n_episodes"]
                self.env_stats[f"{prefix}{k}_mean"] = stat_value
                
                # Print stats if no logger and important metric
                if self.logger is None and k in ["ep_length", "battle_won"]:
                    print(f"{prefix.capitalize()}{k} (mean): {stat_value:.3f}")
        
        # Clear stats after calculating
        stats.clear()
        
    def get_stats(self):
        stats = {}
        stats.update(getattr(self, "return_stats", {}))
        stats.update(getattr(self, "env_stats", {}))
        return stats