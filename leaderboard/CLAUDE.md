# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the Hallucinations Leaderboard - a comprehensive evaluation platform for tracking and ranking hallucinations in Large Language Models across diverse benchmarks. The system consists of a Gradio web application, a backend evaluation worker, and CLI tools for data management.

## Architecture

### Core Components

**Frontend (Gradio Web App)**
- `app.py`: Main Gradio application serving the leaderboard interface
- `src/display/`: UI components, styling, and display utilities
- `src/populate.py`: Data population and leaderboard generation logic

**Backend Evaluation System**
- `backend-cli.py`: Main evaluation worker with dual modes:
  - **Standard mode**: Processes evaluation queue from HuggingFace Hub
  - **Local mode**: Direct evaluation of specified HF models
- `src/backend/run_eval_suite.py`: Core evaluation engine using lm-evaluation-harness
- `src/backend/tasks/`: Custom evaluation tasks (XSum, CNN/DM, HaluEval, FEVER, etc.)

**Queue-Based Architecture**
- **Requests Repository**: `hallucinations-leaderboard/requests` - submission queue
- **Results Repository**: `hallucinations-leaderboard/results` - evaluation outputs
- **Leaderboard Repository**: `hallucinations-leaderboard/leaderboard` - processed results

**CLI Tools Directory (`cli/`)**
- `submit-cli.py`: Submit models for evaluation
- `analysis-cli.py`: Generate analysis reports
- `completed-cli.py`: Manage completed evaluations
- Task-specific upload scripts: `halueval-upload-cli.py`, `fever-upload-cli.py`, etc.

### Evaluation Tasks

The system supports 20+ evaluation benchmarks defined in `src/backend/envs.py`:
- **Knowledge QA**: NQ Open, TriviaQA, PopQA
- **Truthfulness**: TruthfulQA (Gen, MC1, MC2)
- **Hallucination Detection**: HaluEval (QA, Dialogue, Summarization)
- **Summarization**: XSum, CNN/DM
- **Fact Checking**: FEVER, SelfCheckGPT
- **Reading Comprehension**: SQuADv2, RACE
- **Instruction Following**: IFEval

## Common Commands

### Code Quality
```bash
make style          # Format code (black + isort + ruff)
make quality        # Check code quality
```

### Running Applications
```bash
# Start Gradio web interface
python app.py

# Start backend evaluation worker (standard queue mode)
python backend-cli.py

# Direct model evaluation (local mode)
python backend-cli.py --model microsoft/DialoGPT-medium
python backend-cli.py --model microsoft/DialoGPT-medium --precision float32

# Submit model for evaluation
cli/submit-cli.py --model_name "model/name" --model_type "🤖 Instruction-tuned"
```

### Data Management
```bash
# Upload task-specific evaluation data
cli/halueval-upload-cli.py
cli/fever-upload-cli.py
cli/shroom-upload-cli.py

# Analysis and management
cli/analysis-cli.py
cli/completed-cli.py
```

### Environment Setup
```bash
pip install -r requirements.txt
```

## Key Configuration

### Environment Variables
- `H4_TOKEN`: HuggingFace API token for repository access
- `HF_HOME`: Cache directory path (defaults to ".")
- `IS_PUBLIC`: Public/private mode toggle

### Repository Settings (`src/envs.py`)
- `REPO_ID`: Main leaderboard repository
- `QUEUE_REPO`: Evaluation requests queue
- `RESULTS_REPO`: Evaluation results storage
- Rate limiting: 5 submissions per week per user

### Device Configuration (`src/backend/envs.py`)
- Auto-detects: CUDA → MPS → CPU
- MPS device issues automatically fall back to CPU
- Memory cleanup between evaluations

## Backend CLI Modes

### Standard Mode (Queue Processing)
- Monitors HuggingFace evaluation queue
- Downloads requests, processes evaluations, uploads results
- Handles repository synchronization and status updates

### Local Mode (Direct Evaluation) 
- Takes HF model name as input: `--model microsoft/DialoGPT-medium`
- Runs all evaluation tasks on specified model
- Saves results locally without uploading to hub
- Configurable precision and model type

## Task Development

### Adding New Evaluation Tasks
1. Create task directory: `src/backend/tasks/new_task/`
2. Implement task class extending lm-evaluation-harness
3. Add task definition to `Tasks` enum in `src/backend/envs.py`
4. Create upload CLI script in `cli/new_task-upload-cli.py`

### Task Structure
Each task includes:
- `task.py`: Main evaluation implementation
- `utils.py`: Helper functions and data processing
- `README.md`: Task documentation
- YAML configuration files for lm-evaluation-harness

## Error Handling

### MPS Device Issues
The backend automatically handles Apple Silicon MPS device limitations:
- Detects "Placeholder storage has not been allocated" errors
- Falls back to CPU computation for incompatible tasks
- Includes memory cleanup between evaluations

### Batch Size Management
- Starts with `batch_size="auto"`
- Falls back to `batch_size=1` if auto-detection fails
- Retries with device fallback if needed

## Data Flow

1. **Submission**: Models submitted via web interface or CLI
2. **Queue**: Requests stored in HuggingFace Hub queue repository
3. **Processing**: Backend worker picks up pending requests
4. **Evaluation**: lm-evaluation-harness runs tasks on model
5. **Storage**: Results uploaded to results repository
6. **Display**: Leaderboard aggregates and displays results

## Repository Integration

The system operates across multiple HuggingFace repositories:
- Uses HuggingFace Hub for distributed queue management
- Automatic repository synchronization and conflict resolution
- Status tracking: PENDING → RUNNING → FINISHED/FAILED
- Rate limiting and user quota management