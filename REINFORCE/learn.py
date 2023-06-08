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
# from torch.optim.lr_scheduler import lr_scheduler
import torchvision
import torch.nn.functional as F
import importlib
import torchvision.transforms as transforms
from PIL import Image
import cv2
import mycnn

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device(device)
print(f"Using {device} device")

models = {"ResNet18": resnet.ResNet18, 
          "ResNet34": resnet.ResNet34, 
          "ResNet50": resnet.ResNet50, 
          "ResNet101": resnet.ResNet101,
          "rnn": rnn.MyRNN,
          "CNN": mycnn.CNN,
          "LSTM": rnn.LSTMLaneFollower}

### Agent Memory ###
class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def __len__(self):
        return len(self.actions)


### Learner Class ###
class Learner:
    def __init__(
        self, model_name, learning_rate, episodes, clip, max_curvature=1 / 8.0, max_std=0.1
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
        self.model_name = model_name
        self.driving_model = models[model_name]()
        # print(self.driving_model)

        # open and write to file to track progress
        now = datetime.datetime.now()
        self.timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"results/{model_name}_results_{self.timestamp}.txt"
        self.f = open(filename, "w") 
        self.f.write("reward\tloss\tsteps\n")
        self.model_name = model_name

        # hyperparameters
        self.learning_rate = learning_rate
        self.max_curvature = max_curvature
        self.max_std = max_std
        self.episodes = episodes
        self.clip = clip


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

    ## Data preprocessing functions ##
    def _preprocess_(self, full_obs):
        # Extract ROI
        i1, j1, i2, j2 = self.camera.camera_param.get_roi()
        obs = full_obs[i1:i2, j1:j2]
        return obs
    
    def _augment_image(self, cropped_obs):
        # Initialize car_lane_image as an RGB image with random pixel values in the range of 0-255
        car_lane_image = cropped_obs
        # Convert the car_lane_image to a PIL Image object
        car_lane_image_pil = Image.fromarray(car_lane_image)
        # Define the image augmentation transforms
        augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally with 50% probability
            transforms.RandomVerticalFlip(p=0.5),    # Randomly flip vertically with 50% probability
            transforms.RandomRotation(15),           # Randomly rotate image by up to 15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly adjust brightness, contrast, saturation, and hue
            transforms.ToTensor()  # Convert image to tensor
        ])
        # Apply the image augmentation transforms to the car lane image
        augmented_image = augmentation_transforms(car_lane_image_pil)
        return augmented_image
    

    def _resize_image(self, img):
        resized_img = cv2.resize(img, (32, 30))
        return resized_img

    def _grab_and_preprocess_obs_(self, augment=True):
        full_obs = self.car.observations[self.camera.name]
        cropped_obs = self._preprocess_(full_obs)
        resized_obs = self._resize_image(cropped_obs)
        if augment:
            augmented_obs = self._augment_image(resized_obs)
            return augmented_obs.permute(1,2,0)
        else:
            resized_obs_torch = resized_obs / 255.0
            return resized_obs, torch.from_numpy(resized_obs_torch).to(torch.float32)

    ### Training step (forward and backpropagation) ###
    def _train_step_(self, optimizer, observations, actions, discounted_rewards):
        optimizer.zero_grad()
        # Forward propagate through the agent network
        prediction = self._run_driving_model_(observations)
        # back propagate
        loss = self._compute_driving_loss_(
            dist=prediction, actions=actions, rewards=discounted_rewards
        )
        loss.backward()
        nn.utils.clip_grad_value_(self.driving_model.parameters(), self.clip)
        optimizer.step()
        return loss.item() 

    ## The self-driving learning algorithm ##
    def _run_driving_model_(self, image):
        single_image_input = len(image.shape) == 3  # missing 4th batch dimension
        if single_image_input:
            image = image.unsqueeze(0)
                
        image = image.permute(0, 3, 1, 2)
        if self.model_name == "LSTM" or self.model_name == "resnet":
            image = image.unsqueeze(0)
        mu, logsigma = self.driving_model(image)
        # mu, logsigma = torch.chunk(distribution, 2, dim=1)
        mu = self.max_curvature * torch.tanh(mu)  # conversion
        sigma = self.max_std * torch.sigmoid(logsigma) + 0.005  # conversion

        pred_dist = dist.Normal(mu, sigma)
        return pred_dist

    def _compute_driving_loss_(self, dist, actions, rewards):
        with torch.enable_grad():
            neg_logprob = -1 * dist.log_prob(actions) #.sum(axis=-1)
            loss = (neg_logprob * rewards).mean()
            return loss

    def learn(self):
        self._vista_reset_()
        optimizer = optim.Adam(self.driving_model.parameters(), lr=self.learning_rate)
        running_loss = 0
        datasize = 0
        # instantiate Memory buffer
        memory = Memory()
        ## Driving training! Main training block. ##
        max_batch_size = 300
        max_reward = float("-inf")  # keep track of the maximum reward acheived during training

        for i_episode in range(self.episodes):
            self.driving_model.eval()  # set to eval mode because we pass in a single image - not a batch
            self._vista_reset_()
            self.world.set_seed(47)
            memory.clear()
            _, observation = self._grab_and_preprocess_obs_(augment=False)     
            steps = 0
            print(f"Episode: {i_episode}")

            while True:
                curvature_dist = self._run_driving_model_(observation)
                curvature_action = curvature_dist.sample()[0, 0]
                # Step the simulated car with the same action
                self._vista_step_(curvature_action)
                np_obs, observation = self._grab_and_preprocess_obs_(augment=False)
                
                q_lat = np.abs(self.car.relative_state.x)
                road_width = self.car.trace.road_width
                z_lat = road_width / 2
                lane_reward = 1 - (q_lat/z_lat)**2
                reward = lane_reward if not self._check_crash_() else 0.0
                # add to memory
                memory.add_to_memory(observation, curvature_action, reward)
                steps += 1
                # is the episode over? did you crash or do so well that you're done?
                if reward == 0.0:
                    self.driving_model.train()  # set to train as we pass in a batch
                    # determine total reward and keep a record of this
                    total_reward = sum(memory.rewards)
                    print(f"steps: {steps}")

                    batch_size = min(len(memory), max_batch_size)
                    i = torch.randperm(len(memory))[:batch_size]

                    batch_observations = torch.stack(memory.observations, dim=0)
                    batch_observations = torch.index_select(batch_observations, dim=0, index=i)

                    batch_actions = torch.stack(memory.actions)
                    batch_actions = torch.index_select(batch_actions, dim=0, index=i)

                    batch_rewards = torch.tensor(memory.rewards)
                    batch_rewards = self._discount_rewards_(batch_rewards)[i]

                    episode_loss = self._train_step_(
                        optimizer,
                        observations=batch_observations,
                        actions=batch_actions,
                        discounted_rewards=batch_rewards
                    )
                    running_loss += episode_loss
                    datasize += batch_size
                    # episodic loss
                    episode_loss = running_loss / datasize
                    # running_loss += episode_loss
                    print(f"loss: {running_loss}\n")
                    
                    # Write reward and loss to results txt file
                    self.f.write(f"{total_reward}\t{running_loss}\t{steps}\n")
                    
                    # reset the memory
                    memory.clear()

                    # Check gradients norms
                    # print the shape of the gradients for each parameter in the model
                    for name, param in self.driving_model.named_parameters():
                        if param.grad is not None:
                            print(f"{name} gradient shape: {param.grad.shape}")
                    # total_norm = 0
                    # for p in self.driving_model.parameters():
                    #     param_norm = p.grad.data.norm(2) # calculate the L2 norm of gradients
                    #     total_norm += param_norm.item() ** 2 # accumulate the squared norm
                    # total_norm = total_norm ** 0.5 # take the square root to get the total norm
                    # print(f"Total gradient norm: {total_norm}")

                    break
    
    def save(self):
        torch.save(self.driving_model.state_dict(), f"models/{self.model_name}_{self.timestamp}_.pth")