# ParControl: Neural Diversity Regularizes Hallucinations

Official implementation of **"Neural Diversity Regularizes Hallucinations in Small Language Models"**

Paper: [arXiv:XXXX.XXXXX](https://arxiv.org)

## Overview

ParControl implements **ND-LoRA (Neural Diversity Low-Rank Adaptation)**, a novel training method that combines stream-specific LoRA adapters with Barlow Twins regularization to reduce hallucinations in small language models. Our approach achieves significant improvements in factuality across multiple benchmarks while maintaining model quality.

### Key Results

- **15-25% reduction** in hallucination rates on TruthfulQA, HaluEval, and FEVER benchmarks
- **Parameter-efficient**: Only 0.5-2% additional parameters compared to base model
- **Causally validated**: Neural diversity (Dspec) causally reduces hallucinations (p < 0.001)

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA or MPS support
- 16GB+ RAM (32GB recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/ParControl-public.git
cd ParControl-public

# Install dependencies
pip install -r requirements.txt

# Initialize ParScale submodule
git submodule update --init --recursive
```

## Quick Start

### Training

```bash
# Train ND-LoRA model with P=4 streams
python train_parscale.py \
  --P=4 \
  --use-stream-lora \
  --orthogonal-lora \
  --bt-normalization-warmup \
  --target-tokens=20_000_000

# Or use Modal for distributed training
modal run train_parscale::modal__nslP4__OptC9
```

### Evaluation

```bash
# Run hallucination benchmarks
cd leaderboard
python backend_cli.py --model YOUR_MODEL_PATH

# Or use evaluation scripts
python eval_experiments.py --checkpoint PATH_TO_CHECKPOINT
python eval_neurodiversity.py --checkpoint PATH_TO_CHECKPOINT
```

## Model Downloads

Pre-trained model checkpoints are available for all configurations reported in the paper:

- **Baselines**: Qwen2.5-0.5B with P=1 (R=32/64/128)
- **ParScale**: P=2/4/8 with shared LoRA and Barlow Twins
- **ND-LoRA**: P=2/4/8 with stream-specific LoRA and optimized regularization
- **Ablations**: Module ablations, architectural variants

See [`utils/model_checkpoints_paper.py`](utils/model_checkpoints_paper.py) for checkpoint paths and configurations.

### Using Model Checkpoints

The `model_checkpoints_paper.py` module provides organized access to all paper-essential model checkpoints:

```python
from utils.model_checkpoints_paper import (
    CORE_CHECKPOINTS,      # Main results (Tables 1, 7, 8, 9)
    ABLATION_CHECKPOINTS,  # Ablation studies (Table 4)
    MODULE_ABLATION_CHECKPOINTS,  # Module ablations (Table 6)
    ALL_CHECKPOINTS,       # Combined dictionary
    MODEL_NAMES,          # Human-readable names
    BASE_CHECKPOINTS      # Base model paths
)

# Access checkpoint paths
checkpoint_path = CORE_CHECKPOINTS["ND-LoRA_P4"]  # S3 path for ND-LoRA P=4 model
model_name = MODEL_NAMES["ND-LoRA_P4"]  # "ND-LoRA (P=4, OptC9)"

# Use with evaluation scripts
python analyze_experiments.py --model-whitelist ParControl/
python eval_experiments.py --checkpoint CHECKPOINT_PATH
```

### Reading Evaluation Results

The `analyze_experiments.py` script can read evaluation results from `evals-*` directories and generate publication-ready plots:

```bash
# Generate analysis plots from evaluation results
python analyze_experiments.py \
  --results-base-path leaderboard \
  --output-dir plots \
  --plot-mode all pub \
  --analysis-mode full \
  --baseline-mode single-stream

# View generated plots
open plots/pub-full-single-stream-relative.png
```

The script automatically:
- Reads from `leaderboard/evals-{analysis_mode}/` directories
- Maps raw S3 checkpoint paths to human-readable model names using `MODEL_NAMES`
- Generates absolute and relative performance heatmaps
- Creates model-level and evaluation-level summary statistics

> **Note**: Checkpoints will be migrated to public hosting soon. Check back for updated URLs.

## Reproducing Paper Results

All experiments in the paper can be reproduced using Modal for distributed execution:

### Core Results (Tables 1, 7, 8, 9)

```bash
# P=1 baselines (parameter-matched)
modal run train_parscale::modal__P1__r32
modal run train_parscale::modal__P1__r64
modal run train_parscale::modal__P1__r128

# ParScale baselines
modal run train_parscale::modal__P2__r32
modal run train_parscale::modal__P4__r64
modal run train_parscale::modal__P8__r128

# ND-LoRA main results (Optuna-optimized)
modal run train_parscale::modal__nslP2__OptC9
modal run train_parscale::modal__nslP4__OptC9
modal run train_parscale::modal__nslP8__OptC9
```

### Ablation Studies (Tables 4, 6)

```bash
# Component ablations
modal run train_parscale::modal__lP4__r64      # ParScale-BT
modal run train_parscale::modal__sP4           # Stream-LoRA
modal run train_parscale::modal__slP4          # Stream-LoRA-BT
modal run train_parscale::modal__nslP4         # ND-LoRA (original HP)

# Module ablations
modal run train_parscale::modal__p4_nOSL_ablation__modules
```

### Evaluation

```bash
# Deep evaluation (N=1024 samples per task)
cd leaderboard
python eval-cli.py --checkpoint CHECKPOINT_PATH --mode deep

# Corruption experiments for causality analysis
python eval_neurodiversity.py \
  --checkpoint CHECKPOINT_PATH \
  --corruption-methods substitute_tokens substitute_streams \
  --n-samples 128
```

## Architecture

### ND-LoRA Components

1. **Parallel Streams (P)**: Multiple computation paths through the model
2. **Stream-Specific LoRA**: Independent low-rank adapters for each stream
3. **Barlow Twins Regularization**: Decorrelation loss to maintain neural diversity
4. **Optimized Hyperparameters**: λ_BT tuned via Optuna for each P value

### Key Hyperparameters

| Parameter | P=2 | P=4 | P=8 |
|-----------|-----|-----|-----|
| LoRA Rank | 16  | 16  | 16  |
| λ_BT      | 0.29| 0.58| 0.13|
| Design Layer | 20 | 20 | 20 |
| LoRA Modules | q,k,v | q,k,v | q,k,v |

## Repository Structure

```
ParControl/
├── train_parscale.py           # Main training script with Modal entrypoints
├── eval_experiments.py          # Hallucination benchmark evaluation
├── eval_neurodiversity.py       # Causality experiments (corruption analysis)
├── ParScale/                    # Core ParScale implementation (submodule)
├── utils/
│   ├── model_checkpoints_paper.py  # Paper-essential model checkpoints
│   ├── model_utils.py           # Model loading and PEFT setup
│   ├── stream_diagnostics.py    # Stream analysis and monitoring
│   └── ...                      # Other utilities
├── leaderboard/                 # Hallucination evaluation framework
│   ├── backend_cli.py           # Evaluation worker
│   ├── app.py                   # Gradio web interface
│   └── src/backend/tasks/       # Custom evaluation tasks
├── paper/                       # LaTeX source for paper
└── docs/                        # Implementation documentation
```

## Modal Integration

This project uses [Modal](https://modal.com) for running experiments and evaluations. Modal entrypoints in `train_parscale.py` allow distributed training across cloud GPUs.

### Setting up Modal

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Run experiment
modal run train_parscale::modal__nslP4__OptC9
```

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{chakrabarti2025neurodiversity,
  title={Neural Diversity Regularizes Hallucinations in Small Language Models},
  author={Chakrabarti, Kushal and Balachundhar, Nirmal},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Base model: [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- Training data: [The Pile](https://pile.eleuther.ai/)
- ParScale architecture adapted from: [cli99/ParScale](https://github.com/cli99/ParScale)
- Evaluation framework: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
