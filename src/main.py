"""MARL framework for training cooperative reinforcement learning algorithms."""
import os
import sys
import time
import yaml
import json
import argparse
import numpy as np
import torch as th
from pathlib import Path
from types import SimpleNamespace

from utils import time_left, time_str
from buffers import ReplayBuffer
from transforms import OneHot
from policy_networks import BasicMAC
from policy_optimizers import ActorCriticLearner, PPOLearner
from runner import ParallelRunner
import gymnasium as gym
from envs import RWAREWrapper


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                f"{config_name}.yaml",
            ),
            "r",
        ) as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.SafeLoader)
            except yaml.YAMLError as exc:
                assert False, f"{config_name}.yaml error: {exc}"
        return config_dict


def recursive_dict_update(d, u):
    """Merge two dictionaries with nested dictionaries."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_command_line_args():
    """Parse command line arguments and prepare configuration."""
    parser = argparse.ArgumentParser(description="MARL Warehouse framework")
    parser.add_argument("--config", type=str, default="default", 
                        help="Main config file name (in config folder)")
    parser.add_argument("--alg", type=str, default="mappo", 
                        help="Algorithm name (corresponds to a config file in config/algs)")
    parser.add_argument("--env", type=str, default="rware", 
                        help="Environment name (corresponds to a config file in config/envs)")
    parser.add_argument("--seed", type=int, default=1, 
                        help="Random seed")
    
    # Add non-default args for command line
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="Use CUDA")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Custom path to config")
    
    # Parse args
    args = parser.parse_args()
    
    # Load default config
    with open(os.path.join(os.path.dirname(__file__), "config", f"{args.config}.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            assert False, f"Default config error: {exc}"

    # Load algorithm and environment configs
    alg_config = _get_config([], "alg", os.path.join("algs"))
    env_config = _get_config([], "env", os.path.join("envs"))
    
    # Override with specific algorithm config
    with open(os.path.join(os.path.dirname(__file__), "config", "algs", f"{args.alg}.yaml"), "r") as f:
        try:
            alg_config = yaml.load(f, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            assert False, f"{args.alg}.yaml error: {exc}"
    
    # Override with specific environment config
    with open(os.path.join(os.path.dirname(__file__), "config", "envs", f"{args.env}.yaml"), "r") as f:
        try:
            env_config = yaml.load(f, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            assert False, f"{args.env}.yaml error: {exc}"
            
    # Update config with algorithm and environment settings
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict = recursive_dict_update(config_dict, env_config)
    
    # Custom config path if provided
    if args.config_path:
        with open(args.config_path, "r") as f:
            try:
                custom_config = yaml.load(f, Loader=yaml.SafeLoader)
                config_dict = recursive_dict_update(config_dict, custom_config)
            except yaml.YAMLError as exc:
                assert False, f"Custom config error: {exc}"
    
    # Add command line arguments to config
    config_dict["seed"] = args.seed
    config_dict["use_cuda"] = args.cuda
    
    # Update config name based on algorithm
    config_dict["name"] = args.alg
    
    return config_dict


def run(config):
    """Main entry point for training and evaluation"""
    # Convert config to args
    args = SimpleNamespace(**config)
    args.device = "cuda" if args.use_cuda and th.cuda.is_available() else "cpu"
    
    # Set defaults for missing config values
    if not hasattr(args, "critic_type"):
        args.critic_type = "central_v"  # Default to centralized value function
    
    if not hasattr(args, "critic_coef"):
        args.critic_coef = 0.5  # Default critic loss coefficient
        
    if not hasattr(args, "entropy_coef"):
        args.entropy_coef = 0.01  # Default entropy coefficient
    
    if not hasattr(args, "unique_token"):
        args.unique_token = f"{args.name}_{args.env_args['key']}_{args.seed}"
        
    if not hasattr(args, "buffer_cpu_only"):
        args.buffer_cpu_only = True
        
    if not hasattr(args, "batch_size_run"):
        args.batch_size_run = args.batch_size
        
    if not hasattr(args, "render"):
        args.render = False
        
    if not hasattr(args, "common_reward"):
        args.common_reward = True
    
    if not hasattr(args, "reward_scalarisation"):
        args.reward_scalarisation = "sum"
    
    # Initialize environment with wrapper
    env = gym.make(config['env_args']['key'])
    env = RWAREWrapper(env)
    
    # Set up runner with a fixed env creation function that doesn't require kwargs
    def make_env():
        return RWAREWrapper(gym.make(config['env_args']['key']))
    
    # Create the runner with the proper env creation function
    runner = ParallelRunner(env_fn=make_env, args=args)
    
    # Get environment info
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    
    # Set up scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    
    # Set up buffer
    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    
    # Set up MAC
    mac = BasicMAC(buffer.scheme, groups, args)
    
    # Set up runner
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    
    # Set up learner
    if args.learner == "actor_critic_learner":
        learner = ActorCriticLearner(mac, buffer.scheme, args)
    elif args.learner == "ppo_learner":
        learner = PPOLearner(mac, buffer.scheme, args)
    else:
        raise ValueError(f"Unsupported learner: {args.learner}")
    
    if args.use_cuda:
        learner.cuda()
    
    # Run training loop
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    
    start_time = time.time()
    last_time = start_time
    
    print(f"Beginning training for {args.t_max} timesteps")
    
    while runner.t_env <= args.t_max:
        # Run episode
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            
            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            
            learner.train(episode_sample, runner.t_env, episode)

            # Display training info from learner
            if hasattr(learner, "get_train_info"):
                train_info = learner.get_train_info()
                if train_info and (runner.t_env - last_log_T) >= args.log_interval:
                    print(f"Train step: {train_info.get('train_step', runner.t_env)}, "
                          f"actor_loss: {train_info.get('actor_loss', 0):.5f}, "
                          f"critic_loss: {train_info.get('critic_loss', 0):.5f}")
        
        # Test if needed
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            print(f"t_env: {runner.t_env} / {args.t_max}")
            print(f"Estimated time left: {time_left(last_time, last_test_T, runner.t_env, args.t_max)}. "
                  f"Time passed: {time_str(time.time() - start_time)}")
            last_time = time.time()
            last_test_T = runner.t_env
            
            for _ in range(max(1, args.test_nepisode // runner.batch_size)):
                runner.run(test_mode=True)
        
        # Save model if needed
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving models to {save_path}")
            learner.save_models(save_path)
        
        episode += args.batch_size_run
        
        # Simple periodic status update
        if (runner.t_env - last_log_T) >= args.log_interval:
            print(f"Episode: {episode}, env steps: {runner.t_env}")
            last_log_T = runner.t_env
    
    runner.close_env()
    print("Finished Training")


def main(config_dict):
    # Fix random seed
    np.random.seed(config_dict["seed"])
    th.manual_seed(config_dict["seed"])
    
    # GPU config
    config_dict["use_cuda"] = config_dict.get("use_cuda", False)
    
    # Setup results directory
    results_path = os.path.join(
        Path(os.path.dirname(os.path.abspath(__file__))).parent, "results"
    )
    os.makedirs(results_path, exist_ok=True)
    config_dict["local_results_path"] = results_path
    
    # Run
    run(config_dict)


if __name__ == "__main__":
    config_dict = parse_command_line_args()
    main(config_dict)
