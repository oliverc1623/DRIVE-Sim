import matplotlib.pyplot as plt
import vista
import os
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import time
import datetime
import resnet
import rnn
from torch.optim.lr_scheduler import StepLR
import torchvision
import torch.nn.functional as F
import importlib
import math

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

models = {"ResNet18": resnet.ResNet18, 
          "ResNet34": resnet.ResNet34, 
          "ResNet50": resnet.ResNet50, 
          "ResNet101": resnet.ResNet101,
          "rnn": rnn.MyRNN}

### Agent Memory ###
class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.deviation = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward, new_deviation):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)
        self.deviation.append(new_deviation)

    def __len__(self):
        return len(self.actions)


### Learner Class ###
class Learner:
    def __init__(
        self, model_name, learning_rate, episodes, max_curvature=1 / 8.0, max_std=0.1
    ) -> None:
        # Set up VISTA simulator
        trace_root = "../trace"
        trace_path = [
            "20210726-154641_lexus_devens_center",
            "20210726-155941_lexus_devens_center_reverse",
            "20210726-184624_lexus_devens_center",
            "20210726-184956_lexus_devens_center_reverse",
        ]
        trace_path = [os.path.join(trace_root, p) for p in trace_path]
        self.world = vista.World(trace_path, trace_config={"road_width": 4})
        self.car = self.world.spawn_agent(
            config={
                "length": 5.0,
                "width": 2.0,
                "wheel_base": 2.78,
                "steering_ratio": 14.7,
                "lookahead_road": True,
            }
        )
        self.camera = self.car.spawn_camera(config={"size": (200, 320)})
        self.display = vista.Display(
            self.world, display_config={"gui_scale": 2, "vis_full_frame": False}
        )

        self.driving_model = models[model_name]()
        print(self.driving_model)

        # open and write to file to track progress
        now = datetime.datetime.now()
        self.timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"results/{model_name}_results_{self.timestamp}.txt"
        self.f = open(filename, "w") 
        self.f.write("reward\trunning_loss\tepisode_loss\tstep\n")
        self.model_name = model_name

        # hyperparameters
        self.learning_rate = learning_rate
        self.max_curvature = max_curvature
        self.max_std = max_std
        self.episodes = episodes

    def _vista_reset_(self):
        self.world.reset()
        self.display.reset()

    def _vista_step_(self, curvature=None, speed=None):
        if curvature is None:
            curvature = self.car.trace.f_curvature(self.car.timestamp)
        if speed is None:
            speed = self.car.trace.f_speed(self.car.timestamp)

        self.car.step_dynamics(action=np.array([curvature, speed]), dt=1 / 15.0)
        self.car.step_sensors()

    ### Reward function ###
    def _normalize_(self, x):
        x -= torch.mean(x)
        x /= torch.std(x)
        # x = torch.clamp(x,min=0)
        return x

    # Compute normalized, discounted, cumulative rewards (i.e., return)
    def _discount_rewards_(self, rewards, gamma=0.95):
        discounted_rewards = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(0, len(rewards))):
            # update the total discounted reward
            R = R * gamma + rewards[t]
            discounted_rewards[t] = R
        return self._normalize_(discounted_rewards)

    # Check if in terminal state
    def _check_out_of_lane_(self):
        distance_from_center = np.abs(self.car.relative_state.x)
        road_width = self.car.trace.road_width
        half_road_width = road_width / 2
        return distance_from_center > half_road_width

    def _check_exceed_max_rot_(self):
        maximal_rotation = np.pi / 10.0
        current_rotation = np.abs(self.car.relative_state.yaw)
        return current_rotation > maximal_rotation

    def _check_crash_(self):
        return (
            self._check_out_of_lane_() or self._check_exceed_max_rot_() or self.car.done
        )
    
    ## Out of lane punishment
    def _out_of_lane_punishment(self):
        distance_from_center = np.abs(self.car.relative_state.x)
        return -distance_from_center

    ## Data preprocessing functions ##
    def _preprocess_(self, full_obs):
        # Extract ROI
        i1, j1, i2, j2 = self.camera.camera_param.get_roi()
        obs = full_obs[i1:i2, j1:j2]

        # Rescale to [0, 1]
        obs = obs / 255.0
        return obs

    def _grab_and_preprocess_obs_(self):
        full_obs = self.car.observations[self.camera.name]
        obs = self._preprocess_(full_obs)
        obs = torch.from_numpy(obs).to(torch.float32)
        return obs

    ### Training step (forward and backpropagation) ###
    def _train_step_(self, optimizer, observations, actions, discounted_rewards):
        optimizer.zero_grad()
        with torch.enable_grad():
            # Forward propagate through the agent network
            prediction = self._run_driving_model_(observations)
            # back propagate
            loss = self._compute_driving_loss_(
                dist=prediction, actions=actions, rewards=discounted_rewards
            )
            loss.backward()
            nn.utils.clip_grad_value_(self.driving_model.parameters(), 5)
            optimizer.step()
        return loss.item()

    def _compute_driving_loss_(self, dist, actions, rewards):
        with torch.enable_grad():
            neg_logprob = -1 * dist.log_prob(actions)
            loss = torch.mean(neg_logprob * rewards)
            return loss

    ## The self-driving learning algorithm ##
    def _run_driving_model_(self, image):
        single_image_input = len(image.shape) == 3  # missing 4th batch dimension
        if single_image_input:
            image = image.unsqueeze(0)

        image = image.permute(0, 3, 1, 2)
        # print(f"input shape: {image.shape}")
        distribution = self.driving_model(image)
        # print(f"raw output distribution: {distribution}")

        mu, logsigma = torch.chunk(distribution, 2, dim=1)
        mu = self.max_curvature * torch.tanh(mu)  # conversion
        sigma = self.max_std * torch.sigmoid(logsigma) + 0.005  # conversion

        pred_dist = dist.Normal(mu, sigma)
        return pred_dist

    def learn(self):
        self._vista_reset_()

        ## Training parameters and initialization ##
        ## Re-run this cell to restart training from scratch ##
        optimizer = optim.Adam(self.driving_model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        running_loss = 0
        datasize = 0
        # instantiate Memory buffer
        memory = Memory()

        ## Driving training! Main training block. ##
        max_batch_size = 300
        max_reward = float("-inf")  # keep track of the maximum reward achieved during training

        # Define the lane center deviation penalty weight
        lane_center_penalty_weight = 0.01

        for i_episode in range(self.episodes):
            self.driving_model.eval()  # set to eval mode because we pass in a single image - not a batch
            # Restart the environment
            self._vista_reset_()
            memory.clear()
            observation = self._grab_and_preprocess_obs_()
            steps = 0
            print(f"Episode: {i_episode}")

            while True:
                curvature_dist = self._run_driving_model_(observation)      
                softened_dist = dist.Normal(curvature_dist.loc, curvature_dist.scale/2)          
                curvature_action = softened_dist.sample()[0,0].item()
                # Step the simulated car with the same action
                self._vista_step_(curvature_action)
                observation = self._grab_and_preprocess_obs_()
                reward = (1.0) if not self._check_crash_() else 0.0
                deviation = np.abs(self.car.relative_state.x) # calculate distance from lane
                # add to memory
                memory.add_to_memory(observation, curvature_action, reward, deviation)
                steps += 1

                if reward == 0.0:
                    self.driving_model.train()  # set to train as we pass in a batch
                    # total_reward = sum(memory.rewards)
                    print(f"steps: {steps-1}")

                    batch_size = min(len(memory), max_batch_size)
                    i = torch.randperm(len(memory))[:batch_size]

                    batch_observations = torch.stack(memory.observations, dim=0)
                    batch_observations = torch.index_select(
                        batch_observations, dim=0, index=i
                    )

                    batch_actions = torch.tensor(memory.actions)
                    batch_actions = torch.index_select(batch_actions, dim=0, index=i)

                    batch_deviations = torch.tensor(memory.deviation)
                    batch_rewards = torch.tensor(memory.rewards)
                    penalized_rewards = batch_rewards - batch_deviations
                    # print(penalized_rewards)
                    total_reward = sum(penalized_rewards)
                    # print(f"total reward: {total_reward}")
                    batch_rewards = self._discount_rewards_(penalized_rewards)
                    # print(batch_rewards)
                    batch_rewards = torch.index_select(batch_rewards, dim=0, index=i)
                    

                    episode_loss = self._train_step_(
                        optimizer,
                        observations=batch_observations,
                        actions=batch_actions,
                        discounted_rewards=batch_rewards
                    )
                    running_loss += episode_loss
                    print(f"episode loss: {episode_loss}\n")
                    # episodic loss
                    print(f"running loss: {running_loss}\n")
                    
                    # Write reward and loss to results txt file
                    self.f.write(f"{total_reward}\t{running_loss}\t{episode_loss}\t{steps}\n")
                    
                    # reset the memory
                    memory.clear()
                    break
    
    def save(self):
        torch.save(self.driving_model.state_dict(), f"models/{self.model_name}_{self.timestamp}_.pth")