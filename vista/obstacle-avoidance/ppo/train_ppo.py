import os
os.environ["DISPLAY"] = ":2"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
sys.path.append("../../")
from VistaMAEnv import VistaMAEnv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('CustomCNN.py'))))
from CustomCNN import CustomCNN
import copy
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import torch

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(device)
print(f"Using {device} device")


def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        # Initialize the simulator
        trace_config = dict(
            road_width=4,
            reset_mode='default',
            master_sensor='camera_front',
        )
        car_config = dict(
            length=5.,
            width=2.,
            wheel_base=2.78,
            steering_ratio=14.7,
            lookahead_road=True,
        )
        camera_config = {'type': 'camera',
                         'name': 'camera_front',
                         'rig_path': './RIG.xml',
                         'optical_flow_root': '../data_prep/Super-SloMo/slowmo',
                         'size': (400, 640)}
        task_config=dict(n_agents=2,
                         mesh_dir="../../carpack01",
                         init_dist_range=[10.0, 20.0],
                         init_last_noise_range=[0.0, 0.0])
        ego_car_config = copy.deepcopy(car_config)
        ego_car_config['lookahead_road'] = True
        trace_root = "../vista_traces"
        trace_path = [
            "20210726-154641_lexus_devens_center", 
            "20210726-155941_lexus_devens_center_reverse", 
            "20210726-184624_lexus_devens_center", 
            "20210726-184956_lexus_devens_center_reverse", 
        ]
        trace_paths = [os.path.join(trace_root, p) for p in trace_path]
        display_config = dict(road_buffer_size=1000, )
        preprocess_config = {
            "crop_roi": True,
            "resize": True,
            "grayscale": True,
            "binary": False
        }
        env = VistaMAEnv(trace_paths = trace_paths, 
               trace_config = trace_config,
               car_config = car_config,
               display_config = display_config,
               preprocess_config = preprocess_config,
               sensors_configs = [camera_config])
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    time.sleep(1)
    return _init

learning_configs = {
    "policy_type": "CustomCnnPolicy",
    "total_timesteps": 100_000,
    "env_id": "VISTA",
    "learning_rate": 0.0003
}

if __name__ == "__main__":
    for i in range(3,5):
        torch.cuda.empty_cache()
        num_cpu = 8 # Number of processes to use
        vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        # Frame-stacking with 4 frames
        vec_env = VecFrameStack(vec_env, n_stack=4)

        # Create log dir
        log_dir = f"tmp_{i}/"
        os.makedirs(log_dir, exist_ok=True)
        vec_env = VecMonitor(vec_env, log_dir, ('out_of_lane', 'exceed_max_rot', 'distance', 'agent_done'))

        model = PPO("CnnPolicy", 
                    vec_env, 
                    learning_rate=0.0003,
                    verbose=1, 
                    device=device
        )
        timesteps = learning_configs['total_timesteps']
        model.learn(
            total_timesteps=timesteps, 
            progress_bar=True
        )

        # Save the agent
        model.save(f"ppo_trial{i}_naturecnn")
    
