import torch.nn as nn


class TcMLPred(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.reg_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):

        h = self.shared(x)

        tc_pred = self.reg_head(h)
        cls_pred = self.cls_head(h)

        return tc_pred, cls_pred