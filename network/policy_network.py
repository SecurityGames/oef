import torch

# Input: state
# Output: strategy distribution over actions


class BehaviorPolicyModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BehaviorPolicyModel, self).__init__()
        self.state = torch.nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_hidden_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    # for training
    def forward(self, state):
        x = torch.relu(self.state(state))
        x = torch.relu(self.linear_hidden(x))
        x = torch.relu(self.linear_hidden_2(x))
        x = self.out(x)
        return x

    # for use
    def step(self, state):
        x = torch.relu(self.state(state))
        x = torch.relu(self.linear_hidden(x))
        x = torch.relu(self.linear_hidden_2(x))
        x = torch.softmax(self.out(x), dim=1)
        return x
