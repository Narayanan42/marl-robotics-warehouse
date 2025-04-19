import time
import logging
import numpy as np
import torch as th
from collections import defaultdict
from typing import Tuple


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device="cpu"):
        # Handle CUDA errors gracefully
        try:
            if device == "cuda" and not th.cuda.is_available():
                print("WARNING: CUDA is not available, falling back to CPU")
                device = "cpu"
            self.mean = th.zeros(shape, dtype=th.float32, device=device)
            self.var = th.ones(shape, dtype=th.float32, device=device)
        except RuntimeError as e:
            print(f"WARNING: Error initializing on {device}, falling back to CPU: {e}")
            device = "cpu"
            self.mean = th.zeros(shape, dtype=th.float32, device=device)
            self.var = th.ones(shape, dtype=th.float32, device=device)
            
        self.count = epsilon

    def update(self, arr):
        arr = arr.reshape(-1, arr.size(-1))
        batch_mean = th.mean(arr, dim=0)
        batch_var = th.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a + m_b + th.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)
        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger
        self.stats = defaultdict(lambda: [])
        self.use_tb = False

    def setup_tb(self, directory_name):
        try:
            from tensorboard_logger import configure, log_value
            configure(directory_name)
            self.tb_logger = log_value
            self.use_tb = True
            
            self.console_logger.info("*******************")
            self.console_logger.info("Tensorboard logging dir:")
            self.console_logger.info(f"{directory_name}")
            self.console_logger.info("*******************")
        except ImportError:
            self.console_logger.warning("Tensorboard logger not installed - won't log to tensorboard")

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))

        # Log to tensorboard if enabled
        if self.use_tb:
            self.tb_logger(key, value, t)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(
            *self.stats["episode"][-1]
        )
        i = 0
        for k, v in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(
                    np.mean([x[1].item() for x in self.stats[k][-window:]])
                )
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)
    
    def finish(self):
        pass


def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")

    return logger


def print_time(start_time, T, t_max, episode, episode_rewards):
    time_elapsed = time.time() - start_time
    T = max(1, T)
    time_left = time_elapsed * (t_max - T) / T
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    last_reward = "N/A"
    if len(episode_rewards) > 5:
        last_reward = "{:.2f}".format(np.mean(episode_rewards[-50:]))
    print("\033[F\033[F\x1b[KEp: {:,}, T: {:,}/{:,}, Reward: {}, \n\x1b[KElapsed: {}, Left: {}\n".format(episode, T, t_max, last_reward, time_str(time_elapsed), time_str(time_left)), " " * 10, end="\r")


def time_left(start_time, t_start, t_current, t_max):
    if t_current >= t_max:
        return "-"
    time_elapsed = time.time() - start_time
    t_current = max(1, t_current)
    time_left = time_elapsed * (t_max - t_current) / (t_current - t_start)
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    return time_str(time_left)


def time_str(s):
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += "{:d} days, ".format(int(days))
    if hours > 0:
        string += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        string += "{:d} minutes, ".format(int(minutes))
    string += "{:d} seconds".format(int(seconds))
    return string


def test_alg_config_supports_reward(args):
    if args.common_reward:
        # all algorithms support common reward
        return True