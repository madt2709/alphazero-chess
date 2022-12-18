from torch import nn


class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.cross = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, y_v, value, y_p, policy):
        value_error = self.mse(y_v, value)
        policy_error = self.cross(y_p, policy)
        return policy_error + value_error
