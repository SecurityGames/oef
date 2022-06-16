import torch

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
        strategy /= strategy.sum()
        return strategy
