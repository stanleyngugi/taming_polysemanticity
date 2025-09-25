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
        'l0_sparsity': [], 'sae_mse': [], 'train_mse': [], 'val_mse': []
    }

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        logging.info(f'Starting seed {seed} with config: init={args.init}, noise={args.noise}, reg={args.reg}, act={args.act}')

        # Generate and move data to device
        train_x, train_y, val_x, val_y = generate_data(cfg, seed=seed)
        train_x, train_y = train_x.to(device), train_y.to(device)
        val_x, val_y = val_x.to(device), val_y.to(device)

        # Train Network
        model = Network(cfg, init_type=args.init, act_type=args.act).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, 
                               weight_decay=cfg.lambda_reg if args.reg == 'l2' else 0)
        criterion = F.mse_loss

        # Register forward hook to extract hidden activations (last hidden before output)
        hidden_acts = {}
        def hook_fn(module, input, output):
            hidden_acts['last_hidden'] = output
        # Assuming the last activation is before the final Linear; index -2 is the last act_fn
        model.network[-2].register_forward_hook(hook_fn)

        for epoch in range(cfg.epochs):
            model.train()
            optimizer.zero_grad()
            # Forward with noise on hidden
            _ = model(train_x)  # Triggers hook
            hidden = hidden_acts['last_hidden']
            noisy_hidden = add_noise(hidden, args.noise, cfg.sigma_noise, device)
            pred = model.network[-1](noisy_hidden)  # Final linear on noisy hidden
            loss = criterion(pred, train_y)
            if args.reg == 'l1':
                l1_norm = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)  # Weights only
                loss += cfg.lambda_reg * l1_norm
            loss.backward()
            optimizer.step()

            # Val MSE every 100 epochs
            if epoch % 100 == 0 or epoch == cfg.epochs - 1:
                model.eval()
                with torch.no_grad():
                    pred_val = model(val_x)
                    val_loss = criterion(pred_val, val_y).item()
                    logging.info(f'Seed {seed}, Epoch {epoch}: Val MSE {val_loss:.4f}')

        # Final train/val MSE
        with torch.no_grad():
            pred_train = model(train_x)
            train_mse = criterion(pred_train, train_y).item()
            pred_val = model(val_x)
            val_mse = criterion(pred_val, val_y).item()
        metrics_dict['train_mse'].append(train_mse)
        metrics_dict['val_mse'].append(val_mse)

        # Extract clean hidden activations for SAE (no noise)
        model.eval()
        with torch.no_grad():
            _ = model(train_x)  # Triggers hook
            acts = hidden_acts['last_hidden'].detach()

        # Train SAE
        sae = SAE(cfg).to(device)
        sae_optimizer = optim.Adam(sae.parameters(), lr=cfg.lr)
        for sae_epoch in range(500):  # Separate epochs for SAE
            sae.train()
            sae_optimizer.zero_grad()
            recon, hidden = sae(acts)
            loss = F.mse_loss(recon, acts) + cfg.lambda_reg * hidden.abs().sum()  # L1 on SAE hidden
            loss.backward()
            sae_optimizer.step()

        # Compute metrics
        seed_metrics = compute_metrics(sae, acts, model, device)
        for key in ['macs', 'calibrated_macs', 'sparsity_w4', 'l0_sparsity', 'sae_mse']:
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
                     f'Train MSE {train_mse:.4f}, Val MSE {val_mse:.4f}')

    # Aggregate and final log/print
    for key, values in metrics_dict.items():
        avg = np.mean(values)
        std = np.std(values)
        logging.info(f'Average {key.capitalize()}: {avg:.4f} +/- {std:.4f}')
        print(f'Average {key.capitalize()}: {avg:.4f} +/- {std:.4f}')

if __name__ == '__main__':
    main()
