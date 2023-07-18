import gym
import numpy as np
import torch
from gym.spaces.box import Box

from wrapper.benchmarks import *
from wrapper.monitor import *
from wrapper.vec_env import VecEnvWrapper
from wrapper.shmem_vec_env import ShmemVecEnv
from wrapper.dummy_vec_env import DummyVecEnv

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

def make_env(env_id, seed, rank, log_dir, allow_early_resets, args):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            if args.envName == 'Physics-v0':
                env = gym.make(args.envName,
                               args = args
                               )

            else:
                assert False

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env
    return _thunk

def make_vec_envs(args,
                  log_dir,
                  allow_early_resets):

    env_name = args.envName
    seed = args.seed
    num_processes = args.num_processes
    device = args.device

    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, args)
        for i in range(num_processes)
    ]

    """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
    """
    if args.envName == 'Physics-v0':
        env = gym.make(args.envName,
                       args = args
                       )
    else:
        assert False

    spaces = [env.observation_space, env.action_space]
    try:
        envs = ShmemVecEnv(envs, spaces, context='fork')
    except ValueError:
        envs = ShmemVecEnv(envs, spaces, context='spawn')
    envs = VecPyTorch(envs, device)

    return envs, spaces, env.obs_len

class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(np.array(obs)).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)

        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(np.array(obs)).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
