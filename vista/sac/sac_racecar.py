import os
os.environ["DISPLAY"] = ":1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.utils import set_random_seed

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('PrioritizedExperienceReplayBuffer.py'))))
from PrioritizedExperienceReplayBuffer import PrioritizedExperienceReplayBuffer

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__=="__main__":
    env_id = "CarRacing-v2"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    model = SAC(
        "CnnPolicy", 
        vec_env, verbose=1, 
        replay_buffer_class=PrioritizedExperienceReplayBuffer,
        replay_buffer_kwargs=dict(
            alpha=0.6
        )
    )
    model.learn(total_timesteps=1_000, log_interval=4, progress_bar=True)
    model.save("sac_racecar")
    
    del model # remove to demonstrate saving and loading
    
    model = SAC.load("sac_racecar")
    
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = vec_env.step(action)
        if terminated or truncated:
            obs = vec_env.reset()
