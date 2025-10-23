# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains ParControl, a local replication implementation of ParScale (Parallel Scaling Law for Language Models). It provides a complete experimental framework for replicating ParScale experiments on Qwen2.5-0.5B with The Pile dataset, optimized for MacBook Pro M3 development.

The repository also includes a Hallucinations Leaderboard implementation in the `leaderboard/` directory - a comprehensive evaluation platform for tracking and ranking hallucinations in Large Language Models across diverse benchmarks.

## Architecture

### Core Structure
```
ParControl/
├── train_parscale.py              # Main training script
├── run_multi_P_experiment.py      # Multi-P experiment orchestrator
├── test_setup.py                  # Setup validation and tests
├── ParScale/                      # Git submodule with core implementation
│   ├── modeling_qwen2_parscale.py
│   ├── configuration_qwen2_parscale.py
│   ├── cost_analysis.py
│   └── parametric_fit.py
├── src/                          # Local ParScale implementation copy
├── utils/                        # Utility modules
│   ├── model_utils.py            # Model loading, PEFT, ParScale setup
│   ├── data_utils.py             # Pile streaming & tokenization
│   ├── logging_setup.py          # Python logging
│   ├── wandb_setup.py            # W&B integration
│   ├── memory_utils.py           # Memory monitoring
│   ├── stream_diagnostics.py     # Stream analysis
│   └── analysis_utils.py         # Results analysis
├── configs/                      # YAML configurations
├── outputs/                      # Experiment outputs
├── analysis/                     # Results analysis
└── leaderboard/                  # Hallucinations Leaderboard
    ├── app.py                    # Gradio web application
    ├── backend-cli.py            # Backend evaluation worker
    ├── src/
    │   ├── backend/              # Evaluation engine with lm-evaluation-harness
    │   ├── display/              # Frontend UI components
    │   ├── leaderboard/          # Results processing
    │   └── submission/           # Model submission handling
    └── cli/                      # CLI tools for data management
```

### Key Components

**Training Framework**: PEFT-based training that freezes the backbone model and only trains prefix tokens + aggregator for memory efficiency on M3 MacBook Pro.

**Stream Diagnostics**: Real-time monitoring of parallel stream usage, entropy tracking, and collapse detection.

**Experiment Orchestration**: Automated multi-P experiment runner with equal token budgets and reproducible seeds.

**Hallucinations Leaderboard**: Queue-based LLM evaluation system using lm-evaluation-harness, supporting custom hallucination detection tasks across multiple benchmarks (XSum, CNN/DM, HaluEval, FEVER, etc.).

## Common Commands

### Quick Start
```bash
# Test setup and dependencies
python test_setup.py

# Single experiment (P=2 with 16M tokens)
python train_parscale.py --training-mode peft --P 2 --target-tokens 16000000

# Multi-P experiment suite
python run_multi_P_experiment.py --P-values 1 2 4 8 --base-config configs/base_config.yaml
```

### Configuration-Based Training
```bash
# Use predefined configurations
python train_parscale.py --config configs/p2_config.yaml
python train_parscale.py --config configs/p8_config.yaml
```

### Cost Analysis
```bash
# Analyze inference cost for different P values
python ParScale/cost_analysis.py --hidden_size 896 --intermediate_size 4864 --P 2 --batch_size 1
python ParScale/cost_analysis.py --hidden_size 896 --intermediate_size 4864 --P 8 --batch_size 1
```

### Results Analysis
```bash
# Generate comprehensive analysis report
python -c "
from utils.analysis_utils import load_experiment_results, generate_report
results = load_experiment_results('./outputs/multi_P_experiment')
generate_report(results, './outputs/analysis', 'ParScale_Replication')
"
```

### Hallucinations Leaderboard Commands
```bash
# Navigate to leaderboard directory
cd leaderboard/

# Code formatting and quality
make style          # Format code (black + isort + ruff)
make quality        # Check code quality

# Run the applications
python app.py              # Start Gradio web interface  
python backend-cli.py      # Start backend evaluation worker

# Submit model evaluations
cli/submit-cli.py --model_name "model/name" --model_type "🤖 Instruction-tuned"

# Upload task-specific data
cli/halueval-upload-cli.py
cli/fever-upload-cli.py
cli/shroom-upload-cli.py

# Analysis and management
cli/analysis-cli.py
cli/completed-cli.py
```

## Key Technical Concepts

### ParScale Implementation
- **P**: Number of parallel computation streams (1, 2, 4, 8)
- **Prefix Tokens**: 48 learnable tokens per stream for input transformation
- **Aggregator**: MLP that combines stream outputs with softmax weights and label smoothing (ε=0.1)
- **PEFT Training**: Only prefix parameters and aggregator are trainable, backbone frozen

### Training Protocol
- **Equal Token Budget**: All P values train on identical number of tokens
- **Reproducible**: Fixed seeds, logged data provenance, deterministic sampling  
- **Memory Optimized**: Gradient accumulation, bf16 precision, streaming data
- **MPS Backend**: Optimized for Apple Silicon (M1/M2/M3)

### Data Pipeline
- **Streaming**: Uses HuggingFace datasets streaming to avoid 800GB download
- **The Pile**: 8 random shards selected with fixed seeds
- **Token Budget**: 12-16M tokens per experiment, 1M validation tokens
- **Reproducibility**: Exact shard IDs and byte ranges logged

### Hallucinations Leaderboard System
- **Evaluation Framework**: Built on lm-evaluation-harness with custom task extensions
- **Queue-Based Processing**: Models submitted to HuggingFace Hub queues, processed asynchronously  
- **Multi-Repository Architecture**: Separate repos for requests, results, and leaderboard data
- **Custom Tasks**: Specialized benchmarks for hallucination detection (XSum, CNN/DM, HaluEval, FEVER, etc.)
- **Request Management**: Status tracking (PENDING → RUNNING → FINISHED/FAILED) with rate limiting

## Dependencies and Setup

Install dependencies:
```bash
pip install -r requirements.txt

# For cost analysis (optional)
git clone https://github.com/cli99/llm-analysis.git
cd llm-analysis && pip install .
```

Key dependencies:
- PyTorch 2.0+ with MPS support
- transformers 4.48.0 (pinned for compatibility)
- PEFT 0.6.0+ for parameter-efficient training
- datasets for streaming Pile data
- wandb for experiment tracking
- psutil for system monitoring

Leaderboard additional dependencies:
- gradio for web interface
- lm-evaluation-harness for model evaluation
- huggingface-hub for repository management
- Various evaluation metrics (rouge-score, bert-score, sacrebleu)

## Configuration System

All experiments use YAML configs in `configs/`:
- `base_config.yaml`: Default hyperparameters
- `p{1,2,4,8}_config.yaml`: P-specific overrides
- `test_config.yaml`: Fast testing configuration

Key config sections:
```yaml
parscale:
  P: 2                    # Number of parallel streams
  prefix_length: 48       # Tokens per stream
  
training:
  tokens_per_run: 16_000_000
  microbatch_size: 1
  gradient_accumulation_steps: 16
  mixed_precision: "bf16"
  
data:
  num_shards: 8
  seed: 42
```

Leaderboard configuration in `leaderboard/src/envs.py`:
- Repository settings: `REPO_ID`, `QUEUE_REPO`, `RESULTS_REPO`
- Authentication: `H4_TOKEN` for HuggingFace API access
- Rate limiting: 5 submissions per week per user
- Task-specific configs in `leaderboard/src/backend/tasks/*/` with YAML definitions

## Development Workflow

### Running Experiments
1. **Validate setup**: `python test_setup.py`
2. **Single run**: Use `train_parscale.py` with specific P value
3. **Full suite**: Use `run_multi_P_experiment.py` for systematic comparison
4. **Monitor**: Check W&B dashboard or `outputs/logs/` for progress
5. **Analyze**: Use `analysis_utils.py` to generate reports

### Memory Management
- Monitor system resources with built-in `MemoryMonitor`
- Reduce `sequence_length` if OOM (1024 → 512)
- Increase `gradient_accumulation_steps` for larger effective batch size
- Use `bf16` precision on MPS backend

### Debugging
- Set `logging.level: "DEBUG"` in config for verbose output
- Use `--wandb-offline` flag for offline logging during debugging
- Check `outputs/logs/` for detailed Python logs with JSON structure
- Run `test_setup.py` to validate all components

### Working with Leaderboard
1. **Setup**: Navigate to `leaderboard/` directory and install requirements
2. **Development**: Use `make style` for formatting, `make quality` for checks
3. **Testing**: Run `python app.py` to start web interface locally
4. **Backend**: Run `python backend-cli.py` to process evaluation queue
5. **Task Development**: Add custom tasks in `src/backend/tasks/` with YAML configs

## Model Usage

### Loading ParScale Models
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.model_utils import convert_to_parscale_model, create_parscale_config

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Convert to ParScale
config = create_parscale_config(P=4, prefix_length=48)
parscale_model = convert_to_parscale_model(model, config)
```

### Using Trained Models
```python
# Load from checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "./outputs/P4_experiment/checkpoints/final", 
    trust_remote_code=True
)

# Generate with P=4 streams
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_new_tokens=50)
```

## Implementation Notes

- **Zero Modifications**: Core ParScale implementation in `ParScale/` and `src/` is never modified
- **Clean Separation**: All experiment framework built around existing implementation
- **Apple Silicon Optimized**: Uses MPS backend with fallbacks for compatibility  
- **Production Ready**: Comprehensive logging, error handling, and resource monitoring
- **Research Reproducible**: All experiments fully deterministic with logged provenance
- **Offensive Coding**: All code should be written offensively (vs. defensively), i.e. fail loudly and quickly if unexpected behavior occurs.
