import os
os.environ["DISPLAY"] = ":4"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('VistaMAEnv.py'))))
from VistaMAEnv import VistaMAEnv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('CustomCNN.py'))))
from CustomCNN import CustomCNN
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('SeqTransformer.py'))))
from SeqTransformer import SeqTransformer

# Standard Torch
import copy
import time
import torch
import numpy as np

# SB3
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecVideoRecorder, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

device = ("cuda:1" if torch.cuda.is_available() else "cpu")
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
        sensors_config = [
            dict(
                type='camera',
                # camera params
                name='camera_front',
                size=(400, 640), # for lighter cnn 
                # rendering params
                use_lighting=False,
            )
        ]
        task_config=dict(n_agents=2,
                         mesh_dir="../carpack01",
                         init_dist_range=[30.0, 35.0],
                         init_last_noise_range=[0.0, 0.0])
        display_config = dict(road_buffer_size=1000, )
        preprocess_config = {
            "crop_roi": True,
            "resize": True,
            "grayscale": True,
            "binary": False,
            "seq": False
        }
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
        env = VistaMAEnv(
            trace_paths=trace_paths,
            trace_config=trace_config,
            car_configs=[car_config] * task_config['n_agents'],
            sensors_configs=[sensors_config] + [[]] *
            (task_config['n_agents'] - 1),
            preprocess_config=preprocess_config,
            task_config=task_config
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(rank+seed)
    return _init

learning_configs = {
    "policy_type": "CustomCnnPolicy",
    "total_timesteps": 100_000,
    "env_id": "VISTA",
    "learning_rate": 0.0003
}

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
)

if __name__ == "__main__":
    for i in range(1,5):
        torch.cuda.empty_cache()
        num_cpu = 8
        vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        vec_env = VecFrameStack(vec_env, n_stack=4)
    
        # Create log dir
        log_dir = f"tmp-oa_{i}/"
        os.makedirs(log_dir, exist_ok=True)
        vec_env = VecMonitor(vec_env, log_dir, ('out_of_lane', 'exceed_max_rot', 'distance', 'agent_done', 'crashed'))

        # The noise objects for TD3
        n_actions = vec_env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
        # policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
        model = TD3(
            "CnnPolicy",
            vec_env,
            learning_rate=0.0003,
            buffer_size=200_000,
            action_noise=action_noise,
            verbose=1,
            train_freq=1024,
            gradient_steps=-1,
            device=device,
        )
        timesteps = learning_configs['total_timesteps']
        model.learn(
            total_timesteps=timesteps, 
            progress_bar=True
        )

        # Save the agent
        model.save(f"td3-trial{i}-lane-follow")
