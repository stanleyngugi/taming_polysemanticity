import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def generate_data(cfg, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate correlated features
    batch_size = cfg.batch_size
    n_features = cfg.n_features
    feature_groups = [[0, 1, 2], [3, 4], [5, 6, 7]]  # Groups with natural co-occurrences
    
    x = torch.zeros(batch_size, n_features)
    
    for batch_idx in range(batch_size):
        active_groups = []
        for group_idx in range(len(feature_groups)):
            if torch.rand(1) > 0.4:  # 60% chance each group activates
                active_groups.append(group_idx)
        
        for group_idx in active_groups:
            group_features = feature_groups[group_idx]
            base_strength = torch.rand(1) * 0.8 + 0.2  # Correlation strength 0.2-1.0
            shared_component = base_strength * torch.randn(1)
            for feat_idx in group_features:
                indep_component = (1 - base_strength) * torch.randn(1)
                x[batch_idx, feat_idx] = shared_component + indep_component
    
    # Apply sparsity mask
    mask = torch.rand_like(x) > 0.7  # 70% sparsity (30% active)
    x = x * mask.float()
    
    # Ensure non-negative
    x = torch.relu(x)
    
    # Compute complex target
    y = complex_target_function(x)
    
    # Train/val split
    val_size = int(batch_size * cfg.val_split)
    train_size = batch_size - val_size
    train_x, val_x = x[:train_size], x[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]
    
    return train_x, train_y, val_x, val_y

def complex_target_function(x):
    batch_size, n_features = x.shape
    y = torch.zeros_like(x)
    
    # Non-linear interactions with cross-group terms
    y[:, 0] = x[:, 0] * x[:, 1] + 0.5 * x[:, 2]
    y[:, 1] = torch.sin(x[:, 0] + x[:, 2]) + x[:, 1]
    y[:, 2] = x[:, 0] ** 2 + x[:, 1] * x[:, 2]
    
    y[:, 3] = x[:, 3] * x[:, 4] + 0.3 * x[:, 5]  # Cross-group
    y[:, 4] = torch.tanh(x[:, 3] + x[:, 4])
    
    y[:, 5] = x[:, 5] * x[:, 6] * x[:, 7]
    y[:, 6] = x[:, 5] + 0.7 * x[:, 7]
    y[:, 7] = x[:, 6] ** 2 + x[:, 5] * x[:, 7]
    
    return y

def add_noise(hidden, noise_type, sigma, device):
    if noise_type == 'bipolar':
        sign = (torch.rand_like(hidden, device=device) > 0.5).float() * 2 - 1
        return hidden + sign * sigma
    elif noise_type == 'positive_kurtosis':
        # Heavy-tailed noise with positive kurtosis (t-dist df=3)
        xi_np = t.rvs(df=3, size=hidden.shape)
        xi = torch.from_numpy(xi_np).float().to(device) * sigma / np.sqrt(3)  # Normalize variance
        return hidden + xi
    return hidden

def compute_metrics(sae, acts, net_model, device, null_samples=50):
    # Primary: MACS on SAE decoder feature directions
    decoder_weights = sae.decoder.weight.data  # Shape (hidden_dim, d_sae)
    norms = torch.norm(decoder_weights, dim=0, keepdim=True) + 1e-8
    normed = decoder_weights / norms
    cos_matrix = torch.mm(normed.T, normed)  # (d_sae, d_sae)
    off_diag = cos_matrix - torch.eye(cos_matrix.shape[0], device=device)
    macs = torch.mean(torch.abs(off_diag)).item()  # Mean abs cos for i≠j
    
    # Optional null calibration
    macs_nulls = []
    for _ in range(null_samples):
        perm_idx = torch.randperm(decoder_weights.shape[1], device=device)
        perm_weights = decoder_weights[:, perm_idx]
        perm_norms = torch.norm(perm_weights, dim=0, keepdim=True) + 1e-8
        perm_normed = perm_weights / perm_norms
        perm_cos = torch.mm(perm_normed.T, perm_normed)
        perm_off = perm_cos - torch.eye(perm_cos.shape[0], device=device)
        macs_nulls.append(torch.mean(torch.abs(perm_off)).item())
    mu_null = np.mean(macs_nulls)
    std_null = np.std(macs_nulls) + 1e-8
    z_macs = (macs - mu_null) / std_null
    calibrated_macs = 1 / (1 + np.exp(-z_macs))  # Sigmoid to [0,1]
    
    # Complementary: Sparsity - ||W||_4^4 on network weights (peakedness)
    all_weights = torch.cat([p.view(-1) for p in net_model.parameters() if p.dim() > 1])
    l4_norm = torch.norm(all_weights, p=4)
    sparsity_w4 = (l4_norm ** 4 / all_weights.numel()).item()  # High = sparse/peaked
    
    # L0 on SAE hidden (avg non-zero per sample)
    _, hidden_sae = sae(acts)
    l0_sparsity = (hidden_sae > 0).float().mean().item()  # Fraction active
    
    # MSE for SAE recon
    recon, _ = sae(acts)
    sae_mse = torch.nn.functional.mse_loss(recon, acts).item()
    
    return {
        'macs': macs,
        'calibrated_macs': calibrated_macs,
        'sparsity_w4': sparsity_w4,
        'l0_sparsity': l0_sparsity,
        'sae_mse': sae_mse,
        'cos_matrix': cos_matrix.cpu().numpy()  # For plotting
    }

def plot_overlap(cos_matrix_np, title, path):
    plt.imshow(np.abs(cos_matrix_np), cmap='Blues', vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    plt.savefig(path)
    plt.close()
