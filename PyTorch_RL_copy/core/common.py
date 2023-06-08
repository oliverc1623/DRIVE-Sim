import torch
from utils import to_device


def estimate_advantages(rewards, masks, values, gamma, tau, device):
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


def calc_advantages(policy_net, s, r, s_prime, gamma, lmbda, done_mask):
    with torch.no_grad():
        td_target = r + gamma * policy_net.v(s_prime) * done_mask
        delta = td_target - policy_net.v(s)
    delta = delta.cpu().numpy()

    advantage_lst = []
    advantage = 0.0
    for delta_t in delta[::-1]:
        advantage = gamma * lmbda * advantage + delta_t[0]
        advantage_lst.append([advantage])
    advantage_lst.reverse()
    advantage = torch.tensor(advantage_lst, dtype=torch.float)
    return advantage, td_target
