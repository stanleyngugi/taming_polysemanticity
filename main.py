import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from models import Config, TemporalNetwork, SAE, TemporalTracker
from utils import PSIComputer, DataGenerator, add_temporal_noise, VisualizationTools

def setup_logging(experiment_name: str) -> logging.Logger:
    """Setup comprehensive logging for temporal experiments"""
    log_dir = Path(f"experiments/{experiment_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file and console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'experiment.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger, log_dir

def train_network_temporal(cfg: Config, data_generator: DataGenerator, model: TemporalNetwork, 
                          device: torch.device, logger: logging.Logger) -> TemporalNetwork:
    """Train network with temporal PSI tracking and interventions"""
    
    # Initialize components
    psi_computer = PSIComputer(cfg)
    tracker = TemporalTracker(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    # Training metrics
    loss_history = []
    
    logger.info(f"Starting temporal training for {cfg.epochs} epochs")
    
    for epoch in range(cfg.epochs):
        model.train()
        
        # Generate temporal data
        x, y, true_labels = data_generator.generate_temporal_data(epoch, seed=42)
        x, y = x.to(device), y.to(device)
        
        # Forward pass with hidden extraction
        optimizer.zero_grad()
        output, hidden = model.forward(x, return_hidden=True)
        
        # Apply temporal noise
        hidden_noisy = add_temporal_noise(hidden, 'positive_kurtosis', cfg.sigma_noise, epoch)
        
        # Compute loss with regularization
        loss = F.mse_loss(output, y)
        
        # Add L2 regularization (found to be better than L1 for polysemanticity)
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss += cfg.lambda_l2 * l2_norm
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Compute PSI at specified intervals
        if epoch % cfg.psi_compute_interval == 0:
            model.eval()
            with torch.no_grad():
                # Train a quick SAE for PSI computation
                sae = train_sae_fast(cfg, hidden.detach(), device)
                
                # Compute PSI scores
                psi_scores = psi_computer.compute_temporal_psi(sae, hidden, x, true_labels)
                
                # Add to tracker
                tracker.add_snapshot(epoch, model, psi_scores, loss.item(), hidden)
                
                logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}, PSI={psi_scores['total_psi']:.4f}")
                
                # Check for interventions
                if epoch >= cfg.intervention_check_interval:
                    intervention_needed = tracker.recommend_intervention()
                    
                    if intervention_needed:
                        psi_before = psi_scores['total_psi']
                        
                        # Apply intervention
                        model.apply_intervention(intervention_needed)
                        logger.info(f"Applied intervention: {intervention_needed} at epoch {epoch}")
                        
                        # Measure immediate effect (simplified)
                        with torch.no_grad():
                            new_hidden = model.get_hidden_activations(x)
                            new_sae = train_sae_fast(cfg, new_hidden, device)
                            new_psi_scores = psi_computer.compute_temporal_psi(new_sae, new_hidden, x, true_labels)
                            psi_after = new_psi_scores['total_psi']
                        
                        # Record intervention effect
                        intervention_effect = {
                            'epoch': epoch,
                            'type': intervention_needed,
                            'psi_before': psi_before,
                            'psi_after': psi_after,
                            'effectiveness': (psi_before - psi_after) / (psi_before + 1e-8)
                        }
                        tracker.intervention_effects.append(intervention_effect)
                        
                        logger.info(f"Intervention effect: PSI {psi_before:.4f} -> {psi_after:.4f}")
            
            model.train()
    
    return model, tracker, loss_history

def train_sae_fast(cfg: Config, activations: torch.Tensor, device: torch.device, epochs: int = 100) -> SAE:
    """Train SAE quickly for PSI computation"""
    sae = SAE(cfg).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=cfg.lr * 2)  # Faster learning for quick training
    
    for _ in range(epochs):
        recon, hidden = sae(activations)
        loss = F.mse_loss(recon, activations) + cfg.lambda_l1 * hidden.abs().mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return sae

def run_ablation_experiment(base_cfg: Config, experiment_config: Dict, 
                          device: torch.device, log_dir: Path, logger: logging.Logger):
    """Run a single ablation experiment"""
    
    # Update config with experiment parameters
    cfg = Config(**{**base_cfg.__dict__, **experiment_config})
    
    # Create model
    model = TemporalNetwork(cfg, 
                           init_type=experiment_config.get('init_type', 'random'),
                           act_type=experiment_config.get('act_type', 'relu')).to(device)
    
    # Create data generator
    data_generator = DataGenerator(cfg)
    
    # Train model
    trained_model, tracker, loss_history = train_network_temporal(cfg, data_generator, model, device, logger)
    
    # Save results
    experiment_name = f"{experiment_config['name']}"
    results_dir = log_dir / experiment_name
    results_dir.mkdir(exist_ok=True)
    
    # Save final results
    final_psi = tracker.psi_trajectory[-1] if tracker.psi_trajectory else 0.0
    results = {
        'experiment_config': experiment_config,
        'final_psi': final_psi,
        'final_loss': loss_history[-1] if loss_history else float('inf'),
        'psi_trajectory': tracker.psi_trajectory,
        'loss_history': loss_history,
        'interventions_applied': len(tracker.intervention_effects),
        'intervention_effectiveness': tracker.get_intervention_effectiveness()
    }
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    VisualizationTools.plot_psi_trajectory(
        tracker.psi_trajectory, 
        trained_model.intervention_history,
        str(results_dir / 'psi_trajectory.png')
    )
    
    VisualizationTools.plot_intervention_effectiveness(
        tracker,
        str(results_dir / 'intervention_effectiveness.png')
    )
    
    # Save tracker data
    tracker.save_trajectory(str(results_dir / 'temporal_data.pt'))
    
    logger.info(f"Experiment {experiment_name} completed: Final PSI={final_psi:.4f}")
    
    return results

def analyze_temporal_results(results_dir: Path, logger: logging.Logger):
    """Analyze results across all experiments"""
    
    all_results = []
    
    # Load all experiment results
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / 'results.json').exists():
            with open(exp_dir / 'results.json', 'r') as f:
                result = json.load(f)
                all_results.append(result)
    
    if not all_results:
        logger.warning("No experiment results found for analysis")
        return
    
    # Analysis 1: Best configurations
    best_psi = min(all_results, key=lambda x: x['final_psi'])
    logger.info(f"Best PSI configuration: {best_psi['experiment_config']['name']} (PSI: {best_psi['final_psi']:.4f})")
    
    # Analysis 2: Intervention effectiveness
    intervention_summary = {}
    for result in all_results:
        for int_type, effectiveness in result['intervention_effectiveness'].items():
            if int_type not in intervention_summary:
                intervention_summary[int_type] = []
            intervention_summary[int_type].append(effectiveness)
    
    logger.info("Intervention effectiveness summary:")
    for int_type, effects in intervention_summary.items():
        avg_effect = np.mean(effects)
        logger.info(f"  {int_type}: {avg_effect:.4f} ± {np.std(effects):.4f}")
    
    # Analysis 3: Temporal patterns
    logger.info("Temporal analysis:")
    for result in all_results:
        psi_traj = result['psi_trajectory']
        if len(psi_traj) > 5:
            initial_psi = np.mean(psi_traj[:3])
            final_psi = np.mean(psi_traj[-3:])
            change = final_psi - initial_psi
            logger.info(f"  {result['experiment_config']['name']}: PSI change = {change:.4f}")
    
    # Generate comparative visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Final PSI comparison
    plt.subplot(2, 2, 1)
    names = [r['experiment_config']['name'] for r in all_results]
    final_psis = [r['final_psi'] for r in all_results]
    
    bars = plt.bar(range(len(names)), final_psis)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Final PSI Score')
    plt.title('Final PSI by Configuration')
    
    # Color bars by performance
    for bar, psi in zip(bars, final_psis):
        if psi < 0.5:
            bar.set_color('green')
        elif psi < 1.0:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Plot 2: PSI trajectories
    plt.subplot(2, 2, 2)
    for result in all_results:
        if result['psi_trajectory']:
            plt.plot(result['psi_trajectory'], label=result['experiment_config']['name'][:15], alpha=0.7)
    plt.xlabel('Training Epochs (×10)')
    plt.ylabel('PSI Score')
    plt.title('PSI Evolution During Training')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Intervention counts
    plt.subplot(2, 2, 3)
    intervention_counts = [r['interventions_applied'] for r in all_results]
    plt.bar(range(len(names)), intervention_counts)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Number of Interventions')
    plt.title('Interventions Applied by Configuration')
    
    # Plot 4: Loss vs PSI
    plt.subplot(2, 2, 4)
    final_losses = [r['final_loss'] for r in all_results]
    plt.scatter(final_losses, final_psis, alpha=0.7)
    for i, name in enumerate(names):
        plt.annotate(name[:10], (final_losses[i], final_psis[i]), fontsize=8)
    plt.xlabel('Final Loss')
    plt.ylabel('Final PSI')
    plt.title('Loss vs PSI Trade-off')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'comparative_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparative analysis saved to {results_dir / 'comparative_analysis.png'}")

def main():
    parser = argparse.ArgumentParser(description='Temporal Polysemanticity Experiment')
    parser.add_argument('--mode', choices=['single', 'ablation', 'analyze'], default='ablation',
                       help='Experiment mode')
    parser.add_argument('--init', default='random', choices=['random', 'orthogonal'])
    parser.add_argument('--noise', default='positive_kurtosis', choices=['none', 'bipolar', 'positive_kurtosis', 'gaussian'])
    parser.add_argument('--reg', default='l2', choices=['l1', 'l2'])
    parser.add_argument('--act', default='gelu', choices=['relu', 'gelu'])
    parser.add_argument('--correlation_schedule', default='increasing', 
                       choices=['static', 'increasing', 'decreasing', 'pulse'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', default='auto')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"temporal_experiment_{timestamp}"
    
    # Setup logging
    logger, log_dir = setup_logging(experiment_name)
    
    # Base configuration
    base_cfg = Config(
        epochs=args.epochs,
        psi_compute_interval=max(1, args.epochs // 20),  # Compute PSI 20 times during training
        intervention_check_interval=max(10, args.epochs // 10)  # Check interventions 10 times
    )
    
    if args.mode == 'single':
        # Single experiment
        experiment_config = {
            'name': f'single_{args.init}_{args.noise}_{args.reg}_{args.act}',
            'init_type': args.init,
            'noise_type': args.noise,
            'reg_type': args.reg,
            'act_type': args.act,
            'correlation_schedule': args.correlation_schedule
        }
        
        results = run_ablation_experiment(base_cfg, experiment_config, device, log_dir, logger)
        logger.info(f"Single experiment completed with PSI: {results['final_psi']:.4f}")
        
    elif args.mode == 'ablation':
        # Full ablation study
        ablation_configs = [
            # Baseline configurations
            {'name': 'baseline_random_l1_relu', 'init_type': 'random', 'reg_type': 'l1', 'act_type': 'relu', 'correlation_schedule': 'static'},
            {'name': 'baseline_random_l2_gelu', 'init_type': 'random', 'reg_type': 'l2', 'act_type': 'gelu', 'correlation_schedule': 'static'},
            
            # Orthogonal initialization effects
            {'name': 'orthogonal_l1_relu', 'init_type': 'orthogonal', 'reg_type': 'l1', 'act_type': 'relu', 'correlation_schedule': 'static'},
            {'name': 'orthogonal_l2_gelu', 'init_type': 'orthogonal', 'reg_type': 'l2', 'act_type': 'gelu', 'correlation_schedule': 'static'},
            
            # Temporal correlation effects
            {'name': 'increasing_correlation', 'init_type': 'orthogonal', 'reg_type': 'l2', 'act_type': 'gelu', 'correlation_schedule': 'increasing'},
            {'name': 'decreasing_correlation', 'init_type': 'orthogonal', 'reg_type': 'l2', 'act_type': 'gelu', 'correlation_schedule': 'decreasing'},
            {'name': 'pulse_correlation', 'init_type': 'orthogonal', 'reg_type': 'l2', 'act_type': 'gelu', 'correlation_schedule': 'pulse'},
            
            # Best practice combination
            {'name': 'optimal_config', 'init_type': 'orthogonal', 'reg_type': 'l2', 'act_type': 'gelu', 'correlation_schedule': 'increasing'},
        ]
        
        logger.info(f"Starting ablation study with {len(ablation_configs)} configurations")
        
        all_results = []
        for config in ablation_configs:
            try:
                result = run_ablation_experiment(base_cfg, config, device, log_dir, logger)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to run experiment {config['name']}: {e}")
                continue
        
        # Analyze results
        analyze_temporal_results(log_dir, logger)
        
    elif args.mode == 'analyze':
        # Analyze existing results
        if log_dir.exists():
            analyze_temporal_results(log_dir, logger)
        else:
            logger.error(f"Results directory {log_dir} not found")
    
    logger.info("Experiment completed successfully!")

if __name__ == '__main__':
    main()
