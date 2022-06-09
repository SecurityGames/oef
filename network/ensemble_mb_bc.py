import torch


class ensemble_model(torch.nn.Module):
    def __init__(self, bc_model, mb_model, hidden_dim, num_actions, device):
        super().__init__()
        self.bc_model = bc_model.eval()
        self.mb_model = mb_model.eval()
        self.liner_layer = torch.nn.Linear(num_actions * 2, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, num_actions)
        self._softmax_layer = torch.nn.Softmax(dim=-1).to(device)

    def forward(self, state):
        y_1 = self.bc_model(state)
        y_2 = self.mb_model.step(state)
        y = torch.cat([y_1, y_2], dim=1)
        y = self.liner_layer(y)
        y = self.output_layer(y)
        strategy = self._softmax_layer(y)
        return strategy


class ensemble_model_2(torch.nn.Module):
    def __init__(self, num_actions, hidden_dim, device):
        super().__init__()
        self.liner_layer = torch.nn.Linear(num_actions * 2, hidden_dim).to(device)
        self.output_layer = torch.nn.Linear(hidden_dim, num_actions).to(device)
        self._softmax_layer = torch.nn.Softmax(dim=-1).to(device)

    def forward(self, y_1, y_2):
        y = torch.cat([y_1, y_2], dim=1)
        y = self.liner_layer(y)
        y = self.output_layer(y)
        strategy = self._softmax_layer(y)
        return strategy


class mix_policy:
    def __init__(self, bc_model, mb_model, bc_weight):
        self.bc_model = bc_model
        self.mb_model = mb_model
        self.bc_weight = bc_weight
        # self.mb_model = mb_model.eval()

    def step(self, state, legal_actions):
        y_1 = self.bc_model.step(state)
        y_2 = self.mb_model.step(state, legal_actions)

        strategy = self.bc_weight * y_1 + (1 - self.bc_weight) * y_2
        # strategy = torch.softmax(strategy, dim=1)

        return strategy
