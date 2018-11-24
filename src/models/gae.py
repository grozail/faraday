def gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step + 1] - values[step]
        gae = delta + gamma * tau * masks[step + 1] * gae
        returns.insert(0, gae + values[step])
    return returns
