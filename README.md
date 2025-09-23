Temporal Polysemanticity Experiment Framework

A comprehensive framework for studying the temporal dynamics of polysemanticity in neural networks, with automatic intervention scheduling and critical period detection.

Overview

This implementation addresses fundamental questions in mechanistic interpretability:
- How do polysemanticity patterns evolve during training?**
- When are interventions most effective?**
- What is the optimal schedule for applying mitigation strategies?**

The framework provides temporal tracking of the Polysemanticity Suppression Index (PSI) with automatic intervention triggers based on detected critical periods.

Key Features

🔄 Temporal Dynamics Tracking
- Per-epoch PSI computation with efficient memory management
- Critical period detection using gradient analysis
- Intervention timing optimization

🎯 Automatic Intervention System
- Threshold-triggered interventions (weight reset, orthogonal reset, noise injection)
- Effectiveness tracking and adaptive strategy selection
- Real-time intervention impact assessment

📊 Comprehensive Analysis
- Multi-scale visualization of PSI evolution
- Intervention effectiveness analysis
- Comparative configuration analysis

💾 Single-GPU Optimized
- Memory-efficient temporal tracking
- Cached null baselines for PSI computation
- Streamlined SAE training for real-time analysis

Installation

```bash
# Clone and setup environment
git clone <repository>
cd temporal-polysemanticity
pip install -r requirements.txt
```

Quick Start

Single Experiment
```bash
python main.py --mode single --init orthogonal --act gelu --reg l2 --correlation_schedule increasing --epochs 200
```

Full Ablation Study
```bash
python main.py --mode ablation --epochs 200
```

Analyze Existing Results
```bash
python main.py --mode analyze
```

Configuration Options

Core Parameters
- `--init`: Initialization method (`random`, `orthogonal`)
- `--act`: Activation function (`relu`, `gelu`)
- `--reg`: Regularization type (`l1`, `l2`)
- `--correlation_schedule`: Data correlation evolution (`static`, `increasing`, `decreasing`, `pulse`)

Temporal Settings
- `--epochs`: Training duration (default: 200)
- PSI computation interval: Automatically set to epochs/20
- Intervention check interval: Automatically set to epochs/10

Understanding the Output

Key Metrics

1. PSI Score: Lower is better (less polysemantic)
   - `< 0.5`: Good interpretability
   - `0.5-1.0`: Moderate polysemanticity  
   - `> 1.0`: High polysemanticity, intervention needed

2. Intervention Effectiveness: PSI reduction after intervention
   - Positive values indicate successful PSI reduction
   - Tracked per intervention type for strategy optimization

Generated Files

```
experiments/temporal_experiment_YYYYMMDD_HHMMSS/
├── experiment.log              # Detailed training log
├── comparative_analysis.png    # Cross-experiment comparison
├── {experiment_name}/
│   ├── results.json           # Quantitative results
│   ├── psi_trajectory.png     # PSI evolution over time
│   ├── intervention_effectiveness.png
│   └── temporal_data.pt       # Raw temporal data
```

Experimental Design

Temporal Data Generation
The framework uses sophisticated data generation with time-varying complexity:
- **Static**: Fixed feature correlations throughout training
- **Increasing**: Gradually introduce feature correlations
- **Decreasing**: Start with high correlations, reduce over time
- **Pulse**: Periodic correlation changes to test adaptation

PSI Computation
Implements proper polysemanticity measurement:
1. **Per-neuron clustering**: Analyzes whether individual neurons respond to multiple distinct concepts
2. **Feature interference**: Measures correlation between different SAE features
3. **Concept mixing**: Tracks how many true concepts each feature responds to
4. **Null calibration**: Z-scores against random baselines for statistical reliability

### Intervention Strategies
- **Weight Reset**: Selectively reset problematic neuron weights
- **Orthogonal Reset**: Re-apply orthogonal initialization
- **Noise Injection**: Add controlled noise to escape local minima

## Advanced Usage

### Custom Configuration
```python
from models import Config

custom_cfg = Config(
    epochs=500,
    psi_compute_interval=20,  # Check PSI every 20 epochs
    psi_intervention_threshold=0.7,  # Trigger interventions above this PSI
    feature_correlation_strength=0.4,  # Base correlation strength
    correlation_schedule='pulse'  # Dynamic correlation pattern
)
```

### Manual Intervention Timing
```python
# In training loop
if epoch == 100:  # Manual intervention at specific epoch
    model.apply_intervention('orthogonal_reset', strength=1.0)
```

### Custom PSI Analysis
```python
from utils import PSIComputer

psi_computer = PSIComputer(cfg)
psi_scores = psi_computer.compute_temporal_psi(sae, activations, input_data, labels)
print(f"PSI Components: {psi_scores}")
```

## Research Applications

### Critical Period Studies
- Identify when networks transition from monosemantic to polysemantic states
- Test intervention timing hypotheses
- Study plasticity windows for interpretability

### Intervention Optimization  
- Compare intervention strategies across different network architectures
- Optimize intervention scheduling for maximum effectiveness
- Study long-term effects of early interventions

### Scalability Research
- Test findings on larger models
- Investigate layer-specific polysemanticity evolution
- Study cross-model intervention transfer

## Performance Considerations

### Single GPU Optimization
- **Memory Management**: Automatic snapshot pruning (max 50 snapshots)
- **Computation Efficiency**: Cached null baselines, streamlined SAE training
- **Storage**: Compressed temporal data with selective sampling

### Recommended Hardware
- **Minimum**: 8GB GPU memory, 16GB RAM
- **Recommended**: 12GB+ GPU memory, 32GB+ RAM for extended experiments

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or snapshot frequency
   python main.py --mode single --epochs 100  # Shorter experiments
   ```

2. **PSI Computation Fails**
   - Usually due to inactive SAE features
   - Framework automatically handles edge cases

3. **Intervention Not Triggering**
   - Check PSI threshold settings
   - Verify intervention check interval

### Debugging
Enable verbose logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Theoretical Background

This implementation builds on several key insights:

1. **Incidental vs. Necessary Polysemanticity**: Distinguishes between polysemanticity from capacity constraints vs. training artifacts
2. **Temporal Dynamics**: Polysemanticity evolution during training offers intervention opportunities
3. **Critical Periods**: Certain training phases are more amenable to interpretability interventions
4. **Intervention Timing**: Strategic timing can be more effective than intervention strength

## Citation

If you use this framework in research, please cite:
```bibtex
@software{temporal_polysemanticity_framework,
  title={Temporal Polysemanticity Experiment Framework},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request with detailed description

## License

[Specify your license here]

---

For questions or issues, please open an issue on the repository or contact the maintainers.
