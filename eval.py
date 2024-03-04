from griffin import GriffinModel
import torch

x = torch.randn(2, 4, 3)
model = GriffinModel(x)
