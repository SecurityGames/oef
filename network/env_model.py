import torch
import numpy as np
from open_spiel.python.rl_environment import TimeStep, StepType


class EnvModel(torch.nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, output_dim):
        super(EnvModel, self).__init__()
        self.state = torch.nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = torch.nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.linear_hidden_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, state, action):
        x = torch.relu(self.state(state))
        x = torch.cat((x, action), dim=1)
        x = torch.relu(self.linear_hidden(x))
        x = torch.relu(self.linear_hidden_2(x))
        x = self.out(x)
        return x


class DynamicModel(object):
    def __init__(
            self,
            # game setting
            state_length,
            env_action_number,
            legal_action_number,
            player_number,
            game_length,

            # train setting
            env_model=None,
            use_round=True,
            use_deterministic_env_model=True,
            device="cuda"):

        self.device = device
        self.game_step = 0
        self.game_length = game_length
        self.state_length = state_length
        self.env_action_number = env_action_number
        self.legal_action_number = legal_action_number
        self.player_number = player_number
        self.model = env_model.to(self.device)
        self.use_round = use_round
        self.use_deterministic_env_model = use_deterministic_env_model

    def use_model(self, time_step, action):
        # data processing
        state = (torch.tensor(time_step.observations["info_state"])).reshape(1, -1).to(self.device)
        one_hot_action = [1 if i == action[0] else 0 for i in range(self.env_action_number)]
        action_tensor = torch.tensor(one_hot_action).unsqueeze(0).to(self.device)

        # get prediction
        y = self.model(state, action_tensor)
        y = y.squeeze(0)
        chance_node = (y[-1] >= 0.5)
        count_number = 0

        while chance_node:
            count_number += 1
            if count_number > 5:
                break

            if self.use_round:
                state = torch.round(torch.abs(y[:self.player_number * self.state_length])).unsqueeze(0).to(self.device)
            else:
                state = y[:self.player_number * self.state_length].unsqueeze(0).to(self.device)

            legal_actions = torch.round(torch.abs(y[self.player_number * self.state_length:
                                                    self.player_number * self.state_length + self.env_action_number]))
            legal_action = []
            for i in range(self.env_action_number):
                if legal_actions[i] == 1.0:
                    legal_action.append(i)
            if not legal_action:
                legal_action = [i for i in range(self.env_action_number)]

            sampled_action = np.random.choice(legal_action)
            one_hot_action = [1 if i == sampled_action else 0 for i in range(self.env_action_number)]
            action_tensor = torch.tensor(one_hot_action).unsqueeze(0).to(self.device)

            y = self.model(state, action_tensor)
            y = y.squeeze(0)
            chance_node = (y[-1] >= 0.5)

        if self.use_round:
            next_state = torch.round(torch.abs(y[:self.player_number * self.state_length])).reshape(self.player_number, -1)
        else:
            next_state = y[:self.player_number * self.state_length].reshape(self.player_number, -1)

        next_state = next_state.tolist()
        legal_actions = torch.round(torch.abs(y[self.player_number * self.state_length:
                                                self.player_number * self.state_length + self.env_action_number])).tolist()

        legal_action = []
        for i in range(self.legal_action_number):
            if legal_actions[i] == 1.0:
                legal_action.append(i)

        # get done and reward
        rewards = y[- (self.player_number + 2): -2].tolist()
        done = (y[-2] >= 0.5)
        return next_state, legal_action, rewards, done

    def step(self, time_step, action):
        if time_step.step_type == StepType.FIRST:
            self.game_step = 0

        next_state, legal_action, rewards, done = self.use_model(time_step, action)

        if done:
            step_type = StepType.LAST
            next_player_id = -4
        else:
            if self.game_step > self.game_length:
                step_type = StepType.LAST
                next_player_id = -4
            else:
                step_type = StepType.MID
                next_player_id = (time_step.observations["current_player"] + 1) % self.player_number
                if not legal_action:
                    legal_action = [i for i in range(self.legal_action_number)]

        observations = {"info_state": next_state, "legal_actions": [legal_action for _ in range(self.player_number)],
                        "current_player": next_player_id, "serialized_state": []}
        self.game_step += 1
        return TimeStep(observations=observations, rewards=rewards, discounts=0.0, step_type=step_type)
