import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import cv2
import torch.nn.functional as F


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
elif(torch.backends.mps.is_available()):
    device = torch.device("mps")
    print("Device set to : " + str(device))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.direction = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.indicators = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.direction[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.indicators[:]

    def __repr__(self):
        return "RolloutBuffer()"
    
    def __str__(self):
        return f"Num actions: {len(self.actions)}\n \
                 Num state: {len(self.states)}\n \
                 Num logprobs: {len(self.logprobs)}\n \
                 Num rewards: {len(self.rewards)}\n \
                 Num state vals: {len(self.state_values)}\n \
                 Num terminals: {len(self.is_terminals)}\n \
                 Num indicators: {len(self.indicators)}"


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # actor conv layers
        self.actor_conv1 = nn.Conv2d(state_dim, 16, 2)
        self.actor_conv2 = nn.Conv2d(16, 32, 2)
        self.actor_conv3 = nn.Conv2d(32, 64, 2)
        
        # actor linear layers
        self.actor_fc1 = nn.Linear(238145, 512)  # Add +1 for the scalar input
        self.actor_fc2 = nn.Linear(512, action_dim)
        
        # critic conv layers
        self.critic_conv1 = nn.Conv2d(state_dim, 16, 2)
        self.critic_conv2 = nn.Conv2d(16, 32, 2)
        self.critic_conv3 = nn.Conv2d(32, 64, 2)
        
        # critic linear layers
        self.critic_fc1 = nn.Linear(238145, 512)  # Add +1 for the scalar input
        self.critic_fc2 = nn.Linear(512, 1)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self, state, scalar):
        raise NotImplementedError
    
    def act(self, state, scalar):
        # actor
        x = F.relu(self.actor_conv1(state))
        x = F.relu(self.actor_conv2(x))
        x = F.relu(self.actor_conv3(x))
        x = torch.flatten(x, 1)  # Flatten the output for the linear layer
        scalar = scalar.view(-1, 1)  # Reshape scalar to [batch_size, 1] if it's not already
        x = torch.cat((x, scalar), 1)  # Concatenate the scalar with the flattened conv output
        x = F.relu(self.actor_fc1(x))
        action_probs = F.softmax(self.actor_fc2(x), dim=-1)[0]

        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # critic
        y = F.relu(self.critic_conv1(state))
        y = F.relu(self.critic_conv2(y))
        y = F.relu(self.critic_conv3(y))
        y = torch.flatten(y, 1)  # Flatten the output for the linear layer
        y = torch.cat((y, scalar), 1)  # Concatenate the scalar with the flattened conv output
        y = F.relu(self.critic_fc1(y))
        state_values = self.critic_fc2(y)

        return action.detach(), action_logprob.detach(), state_values.detach()
    
    def evaluate(self, state, action, scalar):
        # actor
        x = F.relu(self.actor_conv1(state))
        x = F.relu(self.actor_conv2(x))
        x = F.relu(self.actor_conv3(x))
        x = torch.flatten(x, 1)  # Flatten the output for the linear layer
        scalar = scalar.view(-1, 1)  # Reshape scalar to [batch_size, 1] if it's not already
        x = torch.cat((x, scalar), 1)  # Concatenate the scalar with the flattened conv output
        x = F.relu(self.actor_fc1(x))
        action_probs = F.softmax(self.actor_fc2(x), dim=-1)
        
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # critic
        y = F.relu(self.critic_conv1(state))
        y = F.relu(self.critic_conv2(y))
        y = F.relu(self.critic_conv3(y))
        y = torch.flatten(y, 1)  # Flatten the output for the linear layer
        y = torch.cat((y, scalar), 1)  # Concatenate the scalar with the flattened conv output
        y = F.relu(self.critic_fc1(y))
        state_values = self.critic_fc2(y)
        
        return action_logprobs, state_values, dist_entropy

class PPOIntrospective:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, teacher = False, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
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

    def select_action(self, state, direction):
        with torch.no_grad():
            state = self.preprocess(state, invert=False).to(device)
            direction = torch.tensor(direction, dtype=torch.float).unsqueeze(0).to(device)
            action, action_logprob, state_val = self.policy_old.act(state, direction)

        self.buffer.states.append(state)
        self.buffer.direction.append(direction)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action, direction, state, action_logprob, state_val

    def update(self, correction):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_direction = torch.squeeze(torch.stack(self.buffer.direction, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_direction)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach()) * correction.to(device) 

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # VF loss
            vf_loss = self.MseLoss(state_values, rewards)
            
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * vf_loss - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
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
        
    def preprocess(self, x, invert):
        if invert:
            x = (255 - x)
        x = cv2.resize(x, (64, 64))
        x = torch.from_numpy(x).float() / 255.0
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            x = x.permute(0, 3, 1, 2).to(device)
        return x

    def update_critic(self, teacher_correction, rolloutbuffer):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rolloutbuffer.rewards), reversed(rolloutbuffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(rolloutbuffer.states, dim=0)).detach().to(device)
        old_direction = torch.squeeze(torch.stack(rolloutbuffer.direction, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(rolloutbuffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(rolloutbuffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(rolloutbuffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_direction)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach()) * teacher_correction.to(device)

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # VF loss
            vf_loss = self.MseLoss(state_values, rewards)
            
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * vf_loss - 0.01 * dist_entropy
            # print(f"teacher PPO loss: {loss.mean()}")
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # clear buffer
        self.buffer.clear()