import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

@dataclass
class Config:
    # Basic architecture
    n_features: int = 8
    hidden_dim: int = 4
    num_layers: int = 3
    d_sae: int = 16
    
    # Training parameters
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 1024
    
    # Regularization
    lambda_l1: float = 0.01
    lambda_l2: float = 0.01
    
    # Noise parameters
    sigma_noise: float = 0.1
    kurtosis_df: int = 3
    
    # Temporal tracking
    psi_compute_interval: int = 10  # Compute PSI every N epochs
    intervention_check_interval: int = 25  # Check for interventions every N epochs
    max_snapshots: int = 50  # Limit memory usage
    
    # Intervention thresholds
    psi_intervention_threshold: float = 0.8
    psi_gradient_threshold: float = 0.05  # Rate of PSI increase
    
    # Data correlation
    feature_correlation_strength: float = 0.3
    correlation_schedule: str = "static"  # "static", "increasing", "decreasing"
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

class TemporalNetwork(nn.Module):
    """Network with explicit temporal tracking and intervention capabilities"""
    
    def __init__(self, cfg: Config, init_type='random', act_type='relu'):
        super().__init__()
        self.cfg = cfg
        self.init_type = init_type
        self.act_type = act_type
        
        # Build network with explicit layer access
        self.input_layer = nn.Linear(cfg.n_features, cfg.hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        act_fn = nn.ReLU() if act_type == 'relu' else nn.GELU()
        
        for _ in range(cfg.num_layers - 2):
            self.hidden_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        
        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.n_features)
        self.activation = act_fn
        
        # Initialize weights
        self._initialize_weights()
        
        # Temporal tracking
        self.training_history = []
        self.intervention_history = []
        self.psi_history = []
        
    def _initialize_weights(self):
        """Initialize weights based on specified method"""
        if self.init_type == 'orthogonal':
            gain = init.calculate_gain('relu' if self.act_type == 'relu' else 'linear')
            for layer in [self.input_layer] + list(self.hidden_layers) + [self.output_layer]:
                if isinstance(layer, nn.Linear):
                    init.orthogonal_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)
        else:
            # Default random initialization
            pass
    
    def forward(self, x, return_hidden=False):
        """Forward pass with optional hidden layer return"""
        # Input layer
        hidden = self.activation(self.input_layer(x))
        
        # Hidden layers
        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))
        
        # Output layer
        output = self.output_layer(hidden)
        
        if return_hidden:
            return output, hidden
        return output
    
    def get_hidden_activations(self, x, layer_idx=-1):
        """Extract activations from specific hidden layer"""
        hidden = self.activation(self.input_layer(x))
        
        for i, layer in enumerate(self.hidden_layers):
            if i == layer_idx and layer_idx >= 0:
                return hidden
            hidden = self.activation(layer(hidden))
        
        return hidden  # Return final hidden if layer_idx is -1 or out of bounds
    
    def apply_intervention(self, intervention_type: str, strength: float = 1.0):
        """Apply intervention to network weights"""
        intervention_record = {
            'type': intervention_type,
            'strength': strength,
            'epoch': len(self.training_history)
        }
        
        if intervention_type == 'weight_reset':
            # Reset weights of neurons with highest polysemanticity
            self._reset_problematic_weights(strength)
        elif intervention_type == 'orthogonal_reset':
            # Re-orthogonalize weight matrices
            self._reorthogonalize_weights()
        elif intervention_type == 'noise_injection':
            # Inject noise into weights
            self._inject_weight_noise(strength)
        
        self.intervention_history.append(intervention_record)
    
    def _reset_problematic_weights(self, strength: float):
        """Reset weights of most polysemantic neurons"""
        # Simple implementation: reset random subset of weights
        with torch.no_grad():
            for layer in [self.input_layer] + list(self.hidden_layers):
                if isinstance(layer, nn.Linear):
                    mask = torch.rand_like(layer.weight) < (strength * 0.1)
                    layer.weight.data[mask] = torch.randn_like(layer.weight.data[mask]) * 0.01
    
    def _reorthogonalize_weights(self):
        """Re-apply orthogonal initialization to weight matrices"""
        gain = init.calculate_gain('relu' if self.act_type == 'relu' else 'linear')
        with torch.no_grad():
            for layer in [self.input_layer] + list(self.hidden_layers):
                if isinstance(layer, nn.Linear):
                    init.orthogonal_(layer.weight, gain=gain)
    
    def _inject_weight_noise(self, strength: float):
        """Inject noise into weight matrices"""
        with torch.no_grad():
            for layer in [self.input_layer] + list(self.hidden_layers):
                if isinstance(layer, nn.Linear):
                    noise = torch.randn_like(layer.weight) * strength * 0.01
                    layer.weight.data += noise

class SAE(nn.Module):
    """Sparse Autoencoder for feature extraction"""
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.hidden_dim, cfg.d_sae, bias=False)
        self.decoder = nn.Linear(cfg.d_sae, cfg.hidden_dim, bias=False)
        
        # Normalize initial weights
        with torch.no_grad():
            self.encoder.weight.div_(torch.norm(self.encoder.weight, dim=0, keepdim=True) + 1e-8)
            self.decoder.weight.div_(torch.norm(self.decoder.weight, dim=1, keepdim=True) + 1e-8)
    
    def forward(self, x):
        # Encode with ReLU activation for sparsity
        hidden = torch.relu(self.encoder(x))
        # Decode
        recon = self.decoder(hidden)
        return recon, hidden
    
    def get_feature_activations(self, x):
        """Get sparse feature activations"""
        with torch.no_grad():
            _, hidden = self.forward(x)
            return hidden.detach()

class TemporalTracker:
    """Tracks training dynamics and PSI evolution"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.snapshots = []
        self.psi_trajectory = []
        self.intervention_effects = []
        
    def add_snapshot(self, epoch: int, model: TemporalNetwork, psi_scores: Dict, 
                    loss: float, activations: torch.Tensor):
        """Add training snapshot for temporal analysis"""
        snapshot = {
            'epoch': epoch,
            'psi_scores': psi_scores.copy(),
            'loss': loss,
            'weight_norms': self._compute_weight_norms(model),
            'activation_stats': self._compute_activation_stats(activations)
        }
        
        self.snapshots.append(snapshot)
        self.psi_trajectory.append(psi_scores.get('total_psi', 0.0))
        
        # Limit memory usage
        if len(self.snapshots) > self.cfg.max_snapshots:
            self.snapshots.pop(0)
            self.psi_trajectory.pop(0)
    
    def _compute_weight_norms(self, model: TemporalNetwork) -> Dict[str, float]:
        """Compute weight norms for each layer"""
        norms = {}
        with torch.no_grad():
            norms['input'] = torch.norm(model.input_layer.weight).item()
            for i, layer in enumerate(model.hidden_layers):
                norms[f'hidden_{i}'] = torch.norm(layer.weight).item()
            norms['output'] = torch.norm(model.output_layer.weight).item()
        return norms
    
    def _compute_activation_stats(self, activations: torch.Tensor) -> Dict[str, float]:
        """Compute activation statistics"""
        with torch.no_grad():
            stats = {
                'mean': torch.mean(activations).item(),
                'std': torch.std(activations).item(),
                'sparsity': (activations == 0).float().mean().item(),
                'max_activation': torch.max(activations).item()
            }
        return stats
    
    def detect_critical_period(self, window_size: int = 5) -> bool:
        """Detect if we're in a critical period for intervention"""
        if len(self.psi_trajectory) < window_size:
            return False
        
        recent_psi = self.psi_trajectory[-window_size:]
        psi_gradient = np.gradient(recent_psi)
        
        # Critical period: rapid increase in PSI
        return np.mean(psi_gradient) > self.cfg.psi_gradient_threshold
    
    def recommend_intervention(self) -> Optional[str]:
        """Recommend intervention type based on trajectory"""
        if len(self.psi_trajectory) < 3:
            return None
        
        current_psi = self.psi_trajectory[-1]
        psi_trend = self.psi_trajectory[-1] - self.psi_trajectory[-3]
        
        if current_psi > self.cfg.psi_intervention_threshold:
            if psi_trend > 0.1:
                return 'weight_reset'  # Aggressive intervention
            else:
                return 'orthogonal_reset'  # Conservative intervention
        elif psi_trend > 0.05:
            return 'noise_injection'  # Preventive intervention
        
        return None
    
    def get_intervention_effectiveness(self, lookback: int = 10) -> Dict[str, float]:
        """Analyze effectiveness of past interventions"""
        effectiveness = {}
        
        for intervention in self.intervention_effects[-lookback:]:
            int_type = intervention['type']
            psi_before = intervention['psi_before']
            psi_after = intervention['psi_after']
            
            if int_type not in effectiveness:
                effectiveness[int_type] = []
            
            # Positive effectiveness = PSI reduction
            eff = (psi_before - psi_after) / (psi_before + 1e-8)
            effectiveness[int_type].append(eff)
        
        # Average effectiveness per intervention type
        return {k: np.mean(v) for k, v in effectiveness.items()}
    
    def save_trajectory(self, path: str):
        """Save temporal data for analysis"""
        data = {
            'snapshots': self.snapshots,
            'psi_trajectory': self.psi_trajectory,
            'intervention_effects': self.intervention_effects
        }
        
        torch.save(data, path)
