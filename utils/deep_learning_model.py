import torch
import torch.nn as nn

class IntentModel(nn.Module):
    def __init__(self):
        super(IntentModel, self).__init__()
        self.fc1 = nn.Linear(384, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = IntentModel()