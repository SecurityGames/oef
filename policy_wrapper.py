import torch
import numpy as np


class cfr_policy:
    def __init__(self, policy, num_actions):
        self._policy = policy
        self.num_actions = num_actions

    # state
    def __call__(self, state):
        strategy_dic = self._policy.action_probabilities(state)
        strategy = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            if i in strategy_dic.keys():
                strategy[i] = strategy_dic[i]
        return strategy


class deep_cfr_policy:
    def __init__(self, policy_model, device):
        self._policy = policy_model.to(device)
        self._softmax_layer = torch.nn.Softmax(dim=-1).to(device)
        self.device = device

    # state
    def __call__(self, state):
        info_state_vector = np.array(state.information_state_tensor())
        if len(info_state_vector.shape) == 1:
            info_state_vector = np.expand_dims(info_state_vector, axis=0)
        info_state_vector = torch.FloatTensor(info_state_vector).to(self.device)

        strategy = self._policy(info_state_vector)
        strategy = self._softmax_layer(strategy).cpu()
        return strategy.squeeze(0).detach().numpy()

    # state is tensor vector
    def step(self, state_vector):
        with torch.no_grad():
            strategy = self._policy(state_vector)
            strategy = self._softmax_layer(strategy)
        return strategy


class random_policy:
    def __init__(self, num_actions):
        self.action_number = num_actions

    # state is list
    def __call__(self, state):
        legal_actions = state.legal_actions()
        strategy = np.zeros(self.action_number)
        strategy[legal_actions] = 1.0 / len(legal_actions)
        return strategy


class psro_policy:
    def __init__(self, policy_list, meta_probability, num_actions):
        self._policy = policy_list
        self.num_actions = num_actions
        self.meta_probability = meta_probability

    # state is list
    def __call__(self, state):
        info_state_vector = np.array(state.information_state_tensor())
        legal_actions = state.legal_actions()

        strategy = np.zeros(self.num_actions)
        for i, p in enumerate(self._policy):
            _, prob = p._epsilon_greedy(info_state_vector, legal_actions, epsilon=0)
            weight_prob = prob * self.meta_probability[i]
            strategy += weight_prob
        strategy /= strategy.sum()

        return strategy

    # state is tensor vector
    def step(self, state_vector, legal_actions):
        strategy = np.zeros(self.num_actions)
        for i, p in enumerate(self._policy):
            probs = np.zeros(self.num_actions)
            q_values = p._q_network(state_vector).detach()[0]
            legal_q_values = q_values[legal_actions]
            action = legal_actions[torch.argmax(legal_q_values)]
            probs[action] = 1.0

            weight_prob = probs * self.meta_probability[i]
            strategy += weight_prob

        strategy /= strategy.sum()
        return torch.tensor(strategy)
