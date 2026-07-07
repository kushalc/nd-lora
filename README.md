<div align="center">
<h1>ND-LoRA: Neural Diversity Low-Rank Adaptation</h1>
<i>Neural Diversity Regularizes Hallucinations in Small Language Models</i>

<br/>

<p align="center">
    🔥&nbsp;<a href="#-key-results">Key Results</a>
    | 💡&nbsp;<a href="https://arxiv.org/abs/2510.20690">Paper (arXiv)</a>
    | 📚&nbsp;<a href="#-citation">Citation</a>
</p>
</div>

## Overview

ND-LoRA implements **Neural Diversity Low-Rank Adaptation**, a novel training method that combines stream-specific LoRA adapters with Barlow Twins regularization to reduce hallucinations in small language models. Our approach achieves significant improvements in factuality and faithfulness across multiple benchmarks while maintaining model quality.

### Key Results

- **15-25% reduction** in hallucination rates on TruthfulQA, HaluEval, and MemoTrap benchmarks
- **Parameter-efficient**: Only 0.5-2% additional parameters compared to base model
- **Causally validated**: Neural diversity causally reduces hallucinations (p < 0.001)

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA or MPS support
- 16GB+ RAM (32GB recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/kushalc/nd-lora.git
cd nd-lora

# Install dependencies (creates a uv-managed environment from pyproject.toml)
uv sync
```

## Quick Start

### Training

```bash
# Train an ND-LoRA experiment from its YAML config (config path is a positional arg).
# This launches the job on Modal, detached (survives disconnect) with streamed logs.
uv run scripts/train_ndlora.py configs/ND-LoRA_P4.yaml
```

### Evaluation

```bash
# Evaluate the paper checkpoints on Modal (mode = full | general | deep | quick)
uv run scripts/eval_experiments.py deep

# Neurodiversity causality experiments (corruption analysis) on Modal
uv run scripts/eval_neurodiversity.py --use-modal --meta-mode dose --corruption-mode stream
```

## Model Downloads

Pre-trained model checkpoints are available for all configurations reported in the paper:

- **Baselines**: Qwen2.5-0.5B with P=1 (R=32/64/128)
- **ParScale**: P=2/4/8 with shared LoRA and Barlow Twins
- **ND-LoRA**: P=2/4/8 with stream-specific LoRA and optimized regularization
- **Ablations**: Module ablations, architectural variants

See [`utils/model_checkpoints.py`](utils/model_checkpoints.py) for checkpoint paths and configurations.

### Using Model Checkpoints

The `model_checkpoints.py` module is the single source of truth for the paper-reproduction checkpoints, keyed by clean model name:

```python
from utils.model_checkpoints import (
    CHECKPOINTS,      # clean name -> loadable S3 checkpoint path (alias: ALL_CHECKPOINTS)
    DISPLAY_NAMES,    # clean name -> human-readable label
    MODEL_NAMES,      # eval-dir name -> "ParControl Q0.5B P=k: <Treatment>" (analysis vocabulary)
    BASE_CHECKPOINTS, # base model HF ids
)

# Access checkpoint paths
checkpoint_path = CHECKPOINTS["ND-LoRA_P4"]   # S3 path for ND-LoRA P=4 model
model_name = DISPLAY_NAMES["ND-LoRA_P4"]      # "ND-LoRA (P=4, OptC9)"

# Use with evaluation / analysis scripts
uv run scripts/analyze_experiments.py --model-whitelist nd-lora/
uv run scripts/eval_experiments.py deep
```

### Reading Evaluation Results

The `analyze_experiments.py` script can read evaluation results from `evals-*` directories and generate publication-ready plots:

```bash
# Generate analysis plots from evaluation results
uv run scripts/analyze_experiments.py \
  --results-base-path outputs \
  --output-dir outputs/plots \
  --plot-mode all pub \
  --analysis-mode full \
  --baseline-mode single-stream

# View generated plots
open outputs/plots/pub-full-single-stream-relative.png
```

The script automatically:
- Reads from `outputs/evals-{analysis_mode}/` directories
- Maps raw S3 checkpoint paths to human-readable model names using `MODEL_NAMES`
- Generates absolute and relative performance heatmaps
- Creates model-level and evaluation-level summary statistics

> **Note**: Checkpoints will be migrated to public hosting soon. Check back for updated URLs.

## Reproducing Paper Results

All experiments in the paper can be reproduced on Modal. Each experiment is a YAML in
[`configs/`](configs/) (named for its checkpoint); the training script takes the config
path positionally and launches the job on Modal (detached, with streamed logs):

```bash
uv run scripts/train_ndlora.py configs/<experiment>.yaml
```

### Core Results (Tables 1, 7, 8, 9)

```bash
# P=1 baselines (parameter-matched)
uv run scripts/train_ndlora.py configs/Qwen2.5-0.5B_P1_R32.yaml
uv run scripts/train_ndlora.py configs/Qwen2.5-0.5B_P1_R64.yaml
uv run scripts/train_ndlora.py configs/Qwen2.5-0.5B_P1_R128.yaml

# ParScale baselines
uv run scripts/train_ndlora.py configs/ParScale_P2_R32.yaml
uv run scripts/train_ndlora.py configs/ParScale_P4_R64.yaml
uv run scripts/train_ndlora.py configs/ParScale_P8_R128.yaml

# ND-LoRA main results (Optuna-optimized)
uv run scripts/train_ndlora.py configs/ND-LoRA_P2.yaml
uv run scripts/train_ndlora.py configs/ND-LoRA_P4.yaml
uv run scripts/train_ndlora.py configs/ND-LoRA_P8.yaml
```

### Ablation Studies (Tables 4, 6)

```bash
# Component ablations
uv run scripts/train_ndlora.py configs/ParScale-BT_P4.yaml       # ParScale-BT
uv run scripts/train_ndlora.py configs/Stream_LoRA_P4.yaml        # Stream-LoRA
uv run scripts/train_ndlora.py configs/Stream_LoRA-BT_P4.yaml     # Stream-LoRA-BT
uv run scripts/train_ndlora.py configs/ND-LoRA_P4_Original.yaml   # ND-LoRA (original HP)

# Module ablations
uv run scripts/train_ndlora.py configs/ND-LoRA_P4_no_attention.yaml
uv run scripts/train_ndlora.py configs/ND-LoRA_P4_no_MLP.yaml
```

### Evaluation

```bash
# Deep evaluation (N=1024 samples per task) on Modal
uv run scripts/eval_experiments.py deep

# Corruption experiments for causality analysis (on Modal)
uv run scripts/eval_neurodiversity.py \
  --use-modal \
  --meta-mode dose \
  --corruption-mode stream \
  --limit 128
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
nd-lora/
├── scripts/                     # Runnable entry points
│   ├── train_ndlora.py          # Training (single YAML-driven Modal entrypoint)
│   ├── eval_experiments.py      # Hallucination benchmark evaluation
│   ├── eval_neurodiversity.py   # Causality experiments (corruption analysis)
│   └── analyze_experiments.py   # Results parsing + publication plots
├── configs/                     # One YAML per paper experiment (named for its checkpoint)
├── utils/
│   ├── model_checkpoints.py     # Paper-repro model checkpoints (single source of truth)
│   ├── model_utils.py           # Model loading and PEFT setup
│   ├── stream_diagnostics.py    # Stream analysis and monitoring
│   └── ...                      # Other utilities
├── adhoc/                       # Figure/table generation scripts
├── outputs/                     # All generated outputs (plots, assets, eval caches)
├── ParScale/                    # Core ParScale implementation (vendored)
├── leaderboard/                 # Hallucination evaluation framework (vendored)
│   ├── backend_cli.py           # Evaluation worker
│   ├── app.py                   # Gradio web interface
│   └── src/backend/tasks/       # Custom evaluation tasks
├── pyproject.toml               # Dependencies + packaging (single source)
└── paper/                       # LaTeX source for paper
```

## Modal Integration

This project uses [Modal](https://modal.com) for distributed execution across cloud GPUs. Every script launches its Modal job programmatically (via `app.run(detach=True)` with streamed logs), so you invoke them all the same way — directly with `uv run` — rather than through `modal run` entrypoints.

### Setting up Modal

```bash
# Modal is installed as part of `uv sync`; authenticate once:
modal token new

# Run experiment
uv run scripts/train_ndlora.py configs/ND-LoRA_P4.yaml
```

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{chakrabarti2025neurodiversity,
  title={Neural Diversity Regularizes Hallucinations in Small Language Models},
  author={Chakrabarti, Kushal and Balachundhar, Nirmal},
  journal={arXiv preprint arXiv:2510.20690},
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
