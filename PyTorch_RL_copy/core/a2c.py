import torch
import torch.nn.functional as F

def a2c_step(policy_net, optimizer_policy, states, actions, advantages, td_target):

    """update policy"""
    log_probs = policy_net.get_log_prob(states, actions)
    policy_loss = -(log_probs * advantages).mean() + F.smooth_l1_loss(policy_net.v(states), td_target)
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()
