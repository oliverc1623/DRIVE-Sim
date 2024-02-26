import torch
from torch.distributions import Bernoulli


def introspect(
    state,
    teacher_source_policy,
    teacher_target_policy,
    t,
    inspection_threshold=0.15,
    introspection_decay=0.999999, # paper's 0.99999
    burn_in=700_000,
):
    h = 0
    probability = introspection_decay**(max(0, t - burn_in))
    p = Bernoulli(probability).sample()
    if t > burn_in and p == 1:
        _, _, teacher_source_val = teacher_source_policy.act(state)
        _, _, teacher_target_val = teacher_target_policy.act(state)
        if abs(teacher_target_val - teacher_source_val) <= inspection_threshold:
            h = 1
    return h

def correct(
    rolloutbuffer,
    student_policy,
    teacher_policy
):
    teacher_ratios = []
    student_ratios = []

    for action, state, indicator, logprob, in zip(rolloutbuffer.actions, rolloutbuffer.states, rolloutbuffer.indicators, rolloutbuffer.logprobs):
        
        if indicator:
            
            # compute importance sampling ratio
            _, student_action_logprob, _ = student_policy.act(state.detach())
            ratio = student_action_logprob / logprob
            
            # append corrections
            teacher_ratios.append(1.0)
            student_ratios.append(torch.clamp(ratio, -2, 2).item())

        else:
            # compute importance sampling ratio
            _, teacher_action_logprob, _ = teacher_policy.act(state.detach())
            ratio = teacher_action_logprob / logprob

            teacher_ratios.append(torch.clamp(ratio, -2, 2).item())
            student_ratios.append(1.0)

    teacher_ratios = torch.tensor(teacher_ratios).float()
    student_ratios = torch.tensor(student_ratios).float()

    return teacher_ratios, student_ratios

