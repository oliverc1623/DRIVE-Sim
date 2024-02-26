import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from torch.autograd import Function
import numpy as np
import cv2
from collections import deque

device = "cuda:0"

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def generate_minibatch(self, minibatch_size=128):
        batch_size = len(self.states)  # Assuming all lists are of the same size
        indices = np.random.choice(batch_size, minibatch_size, replace=False)
        
        minibatch_states = [self.states[i] for i in indices]
        minibatch_actions = [self.actions[i] for i in indices]
        minibatch_logprobs = [self.logprobs[i] for i in indices]
        minibatch_rewards = [self.rewards[i] for i in indices]
        minibatch_state_values = [self.state_values[i] for i in indices]
        minibatch_is_terminals = [self.is_terminals[i] for i in indices]
        
        return minibatch_states, minibatch_actions, minibatch_logprobs, minibatch_rewards, minibatch_state_values, minibatch_is_terminals


class Encoder(nn.Module):
    def __init__(self, latent_size = 32, input_channel = 3):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )
        self.linear_mu = nn.Linear(2*2*256, latent_size)

    def forward(self, x):
        x = self.main(x/255.0)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        return mu


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, latent_size):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(latent_size*4, 64),
                        nn.ReLU(),
                        nn.Linear(64, 256),
                        nn.ReLU(),
                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.Linear(512, action_dim),
                        nn.Softmax(-1)
                    )
        # critic
        self.critic = nn.Sequential(
                            nn.Linear(latent_size*4, 64),
                            nn.ReLU(),
                            nn.Linear(64, 256),
                            nn.ReLU(),
                            nn.Linear(256, 512),
                            nn.ReLU(),
                            nn.Linear(512, 1)
                        )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, features):
        action_probs = self.actor(features)[0]
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(features)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, features, action):        
        action_probs = self.actor(features)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, latent_size=64):

        self.main = Encoder(latent_size=latent_size).to(device)

        # saved checkpoints could contain extra weights such as linear_logsigma 
        weights = torch.load("../../LUSR/checkpoints/encoder2.pt", map_location=torch.device('cpu'))
        for k in list(weights.keys()):
            if k not in self.main.state_dict().keys():
                del weights[k]
        self.main.load_state_dict(weights)
        print("Loaded Weights")

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        # stacked buffer
        self.feature_queue = deque(maxlen=4)
        blank1 = self.preprocess(np.zeros((64,64,3)))
        blank2 = self.preprocess(np.zeros((64,64,3)))
        blank3 = self.preprocess(np.zeros((64,64,3)))
        self.feature_queue.append(blank1)
        self.feature_queue.append(blank2)
        self.feature_queue.append(blank3)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, latent_size).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, latent_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        with torch.no_grad():
            feature = self.preprocess(state).to(device)
            self.feature_queue.append(feature)
            stacked_features = torch.cat(list(self.feature_queue), 1)
            action, action_logprob, state_val = self.policy_old.act(stacked_features.detach())

        self.buffer.states.append(stacked_features)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward * 0.95)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # value function loss
            vf_loss = self.MseLoss(state_values, rewards)
            
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 1.0 * vf_loss - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
    def preprocess(self, x, invert=False):
        if invert:
            x = (255 - x)
        # start_row, end_row = (4, 68)
        # start_col, end_col = (4, 68)
        # x = x[start_row:end_row, start_col:end_col, :]
        x = torch.from_numpy(x).float()
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            x = x.permute(0, 3, 1, 2).to(device)

        features = self.main(x.float())
        return features
