import torch
import torch.nn as nn
import torch.nn.functional as F


class BirdMoE(nn.Module):
    def __init__(self, input_dim=1536, num_classes=234, num_experts=4):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, num_classes),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        route_weights = F.softmax(self.router(x), dim=1)
        expert_preds = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(route_weights.unsqueeze(-1) * expert_preds, dim=1)
