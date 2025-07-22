import torch
from torch import nn
import torch.nn.init as init
import copy
import random
from pygame import time

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.network = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Tanh()
    )
    for m in self.modules():
      if isinstance(m, torch.nn.Linear):
        # Normal distribution with higher std
        init.normal_(m.weight)
        init.constant_(m.bias, 0.0)  # Optional: keep bias zero to avoid drift
    self.to(device)
  def forward(self, x):
    value =  self.network(x) * 10
    return value
def clone_model(model):
    return copy.deepcopy(model).to(device)

def mutate_model(model, mutation_rate=0.4, mutation_strength=1.0):
    new_model = clone_model(model)
    with torch.no_grad():
        for param in new_model.parameters():
            if torch.rand(1).item() < mutation_rate:
                noise = torch.randn_like(param) * mutation_strength
                param.add_(noise)
    return new_model