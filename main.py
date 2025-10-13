import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import numpy as np
from models import Config, Network, SAE
from utils import generate_data, add_noise, compute_metrics, plot_overlap

def main():
    parser = argparse.ArgumentParser(description='Polysemanticity Experiment Ablation Runner')
    parser.add_argument('--init', default='random', choices=['random', 'orthogonal'], help='Initialization type')
    parser.add_argument('--noise', default='bipolar', choices=['none', 'bipolar', 'positive_kurtosis'], help='Noise type')
    parser.add_argument('--reg', default='l1', choices=['l1', 'l2'], help='Regularization type')
    parser.add_argument('--act', default='relu', choices=['relu', 'gelu'], help='Activation type')
    parser.add_argument('--seeds', type=int, default=10, help='Number of random seeds to run')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(filename='experiment.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    cfg = Config()

    # Lists to collect metrics across seeds
    metrics_dict = {
        'macs': [], 'calibrated_macs': [], 'sparsity_w4': [], 
        'l0_sparsity': [], 'sae_mse': [], 'train_mse': [], 'val_mse': [],
        'interference': []  # Added for weighted polysemanticity measure
    }

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        logging.info(f'Starting seed {seed} with config: init={args.init}, noise={args.noise}, reg={args.reg}, act={args.act}')

        # Generate data with importances
        train_x, train_y, val_x, val_y, importances = generate_data(cfg, seed=seed)
        train_x, train_y = train_x.to(device), train_y.to(device)
        val_x, val_y = val_x.to(device), val_y.to(device)

        # Train Network with increased epochs
        model = Network(cfg, init_type=args.init, act_type=args.act).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, 
                               weight_decay=cfg.lambda_reg if args.reg == 'l2' else 0)
        criterion = F.mse_loss

        # Forward hook for hidden acts
        hidden_acts = {}
        def hook_fn(module, input, output):
            hidden_acts['last_hidden'] = output
        model.network[-2].register_forward_hook(hook_fn)  # Last act before output

        for epoch in range(2000):  # Increased epochs for better convergence
            model.train()
            optimizer.zero_grad()
            _ = model(train_x)
            hidden = hidden_acts['last_hidden']
            noisy_hidden = add_noise(hidden, args.noise, cfg.sigma_noise, device)
            pred = model.network[-1](noisy_hidden)
            loss = criterion(pred, train_y)
            if args.reg == 'l1':
                l1_norm = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)
                loss += cfg.lambda_reg * l1_norm
            loss.backward()
            optimizer.step()

            # Monitor val MSE and dead neurons more frequently
            if epoch % 50 == 0 or epoch == 1999:  # Every 50 epochs
                model.eval()
                with torch.no_grad():
                    pred_val = model(val_x)
                    val_loss = criterion(pred_val, val_y).item()
                    # Check dead neurons in network (fraction zero across batch)
                    zero_frac = (hidden == 0).float().mean().item()
                    logging.info(f'Seed {seed}, Epoch {epoch}: Val MSE {val_loss:.4f}, Net Dead Frac {zero_frac:.4f}')

        # Final MSEs
        with torch.no_grad():
            pred_train = model(train_x)
            train_mse = criterion(pred_train, train_y).item()
            pred_val = model(val_x)
            val_mse = criterion(pred_val, val_y).item()
        metrics_dict['train_mse'].append(train_mse)
        metrics_dict['val_mse'].append(val_mse)

        # Extract clean acts
        model.eval()
        with torch.no_grad():
            _ = model(train_x)
            acts = hidden_acts['last_hidden'].detach()

        # Train SAE with increased epochs and monitoring
        sae = SAE(cfg).to(device)
        sae_optimizer = optim.Adam(sae.parameters(), lr=cfg.lr)
        for sae_epoch in range(1000):  # Increased for SAE convergence
            sae.train()
            sae_optimizer.zero_grad()
            recon, hidden = sae(acts)
            loss = F.mse_loss(recon, acts) + cfg.lambda_reg * hidden.abs().sum()
            loss.backward()
            sae_optimizer.step()

            # Monitor SAE L0/dead every 100 epochs
            if sae_epoch % 100 == 0 or sae_epoch == 999:
                with torch.no_grad():
                    sae_l0 = (hidden > 0).float().mean().item()
                    logging.info(f'Seed {seed}, SAE Epoch {sae_epoch}: L0 {sae_l0:.4f}')

        # Compute metrics with importances
        seed_metrics = compute_metrics(sae, acts, model, device, importances, null_samples=200)
        for key in metrics_dict.keys():
            if key in seed_metrics:
                metrics_dict[key].append(seed_metrics[key])

        # Plot and log
        plot_overlap(seed_metrics['cos_matrix'], 
                     f'Cosine Overlap: {args.init}_{args.noise}_{args.reg}_{args.act}_seed{seed}', 
                     f'overlap_{args.init}_{args.noise}_{args.reg}_{args.act}_seed{seed}.png')
        logging.info(f'Seed {seed} Metrics: MACS {seed_metrics["macs"]:.4f}, '
                     f'Calib MACS {seed_metrics["calibrated_macs"]:.4f}, '
                     f'Sparsity W4 {seed_metrics["sparsity_w4"]:.4f}, '
                     f'L0 Sparsity {seed_metrics["l0_sparsity"]:.4f}, '
                     f'SAE MSE {seed_metrics["sae_mse"]:.4f}, '
                     f'Interference {seed_metrics["interference"]:.4f}, '
                     f'Train MSE {train_mse:.4f}, Val MSE {val_mse:.4f}')

    # Aggregate results
    for key, values in metrics_dict.items():
        avg = np.mean(values)
        std = np.std(values)
        logging.info(f'Average {key.capitalize()}: {avg:.4f} +/- {std:.4f}')
        print(f'Average {key.capitalize()}: {avg:.4f} +/- {std:.4f}')

if __name__ == '__main__':
    main()
