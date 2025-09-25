import torch
import torch.nn as nn
import torch.nn.init as init
from dataclasses import dataclass

@dataclass
class Config:
    n_features: int = 8
    hidden_dim: int = 12  # Overcapacity: greater than n_features to allow for disentangled representations
    num_layers: int = 5
    lr: float = 1e-3
    epochs: int = 1000
    lambda_reg: float = 0.01
    sigma_noise: float = 0.2
    d_sae: int = 16  # Overcomplete for SAE to extract more features than hidden_dim
    batch_size: int = 1024
    val_split: float = 0.2  # Fraction for validation split in data generation

class Network(nn.Module):
    def __init__(self, cfg: Config, init_type: str = 'random', act_type: str = 'relu'):
        super().__init__()
        self.cfg = cfg
        # Select activation function
        if act_type == 'relu':
            act_fn = nn.ReLU()
            gain_type = 'relu'  # Supported by calculate_gain
        elif act_type == 'gelu':
            act_fn = nn.GELU()
            gain_type = 'relu'  # Approximate GELU with ReLU gain (common practice as GELU is not directly supported)
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

        # Build layers
        layers = [nn.Linear(cfg.n_features, cfg.hidden_dim)]
        layers.append(act_fn)
        for _ in range(cfg.num_layers - 2):
            layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(cfg.hidden_dim, cfg.n_features))
        self.network = nn.Sequential(*layers)

        # Initialization
        if init_type == 'orthogonal':
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    gain = init.calculate_gain(gain_type)
                    init.orthogonal_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)
        elif init_type != 'random':
            raise ValueError(f"Unsupported init type: {init_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = nn.Linear(cfg.hidden_dim, cfg.d_sae, bias=False)
        self.decoder = nn.Linear(cfg.d_sae, cfg.hidden_dim, bias=False)
        # Normalize decoder weights to unit norm per column (feature directions)
        with torch.no_grad():
            self.decoder.weight.div_(torch.norm(self.decoder.weight, dim=0, keepdim=True) + 1e-8)  # Avoid div by zero

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = nn.functional.relu(self.encoder(x))
        recon = self.decoder(hidden)
        return recon, hidden
