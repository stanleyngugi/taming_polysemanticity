import torch
import torch.nn as nn
import torch.nn.init as init
from dataclasses import dataclass

@dataclass
class Config:
    n_features: int = 8
    hidden_dim: int = 12  # Overcapacity: greater than n_features for incidental polysemanticity study
    num_layers: int = 5
    lr: float = 1e-3
    epochs: int = 1000
    lambda_reg: float = 1e-4  # Lowered based on research to prevent over-regularization and dead latents
    sigma_noise: float = 0.1  # Adjusted lower to reduce disruption in training
    d_sae: int = 16  # Overcomplete for SAE to disentangle features
    batch_size: int = 1024
    val_split: float = 0.2

class Network(nn.Module):
    def __init__(self, cfg: Config, init_type: str = 'random', act_type: str = 'relu'):
        super().__init__()
        self.cfg = cfg
        # Activation selection with research-based choice: GELU preferred over ReLU for smoother gradients and avoiding dying neurons
        if act_type == 'relu':
            act_fn = nn.ReLU()
            gain_type = 'relu'
        elif act_type == 'gelu':
            act_fn = nn.GELU()
            gain_type = 'relu'  # Approximation as GELU not directly supported; research shows this works well
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

        # Layer construction: Input to hidden, intermediate hiddens, output
        layers = [nn.Linear(cfg.n_features, cfg.hidden_dim)]
        layers.append(act_fn)
        for _ in range(cfg.num_layers - 2):
            layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(cfg.hidden_dim, cfg.n_features))
        self.network = nn.Sequential(*layers)

        # Initialization: Orthogonal for preserving norms and minimizing initial correlations (key for polysemanticity toys)
        if init_type == 'orthogonal':
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    gain = init.calculate_gain(gain_type)
                    init.orthogonal_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)  # Zero biases standard, but research suggests small positive for ReLU to avoid initial death
        elif init_type != 'random':
            raise ValueError(f"Unsupported init type: {init_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = nn.Linear(cfg.hidden_dim, cfg.d_sae, bias=True)  # Add bias for positive shifts to prevent dying activations
        self.decoder = nn.Linear(cfg.d_sae, cfg.hidden_dim, bias=False)
        
        # Research-based init: Encoder as transpose of decoder to prevent dead latents (from OpenAI SAE scaling paper)
        with torch.no_grad():
            self.encoder.weight.data = self.decoder.weight.data.t()
            # Normalize decoder columns to unit norm (standard for feature directions in SAEs)
            self.decoder.weight.div_(torch.norm(self.decoder.weight, dim=0, keepdim=True) + 1e-8)
            # Positive bias init for encoder to encourage initial activations >0
            init.uniform_(self.encoder.bias, 0.0, 0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Use LeakyReLU instead of ReLU to fix dying neuron problem (allows small negative gradients)
        hidden = nn.functional.leaky_relu(self.encoder(x), negative_slope=0.01)
        recon = self.decoder(hidden)
        return recon, hidden
