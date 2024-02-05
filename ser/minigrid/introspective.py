from torch.distributions import Bernoulli

def introspect(
    state,
    teacher_source_policy,
    teacher_target_policy,
    t,
    inspection_threshold=0.9,
    introspection_decay=0.99999,
    burn_in=0,
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

    for action, state, indicator, in zip(rolloutbuffer.actions, rolloutbuffer.states, rolloutbuffers.indicators):
        if indicator:

            # compute importance sampling ratio
            student_action_logprob, _, _ = student_policy.evaluate(state, action)
            teacher_action_logprob, _, _ = teacher_policy.evaluate(state, action)
            ratio = student_action_logprob / teacher_action_logprob

            # append corrections
            teacher_ratios.append(1)
            student_ratios.append(ratio)

        else:

            # compute importance sampling ratio
            student_action_logprob, _, _ = student_policy.evaluate(state, action)
            teacher_action_logprob, _, _ = teacher_policy.evaluate(state, action)
            ratio = teacher_action_logprob / student_action_logprob

            teacher_ratios.append(ratio)
            student_ratios.append(1)

    return teacher_ratios, student_ratios
            