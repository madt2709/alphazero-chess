from torch import nn


class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.cross = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, y_p, policy, y_v, value):
        value_error = self.mse(y_v.view(-1), value.view(-1))
        policy_error = self.cross(y_p.view(-1), policy.view(-1))
        return policy_error + value_error
