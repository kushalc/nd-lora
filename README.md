<div align="center">
<h1>ND-LoRA</h1>
<h3>Neural Diversity Regularizes Hallucinations in Language Models</h3>

<i>Decorrelated parallel representations are a provable, near-zero-cost mechanism<br/>
for reducing hallucination at a fixed parameter and data budget.</i>

<br/><br/>

<p align="center"><b>Published in Transactions on Machine Learning Research (TMLR), 2026</b></p>

<p align="center">
    📄&nbsp;<a href="https://openreview.net/pdf?id=5l9ZflyApA">Paper (TMLR)</a>
    &nbsp;·&nbsp; 🔬&nbsp;<a href="#results">Results</a>
    &nbsp;·&nbsp; ⚡&nbsp;<a href="#quickstart">Quickstart</a>
    &nbsp;·&nbsp; 🔁&nbsp;<a href="#reproducing-the-paper">Reproduce</a>
    &nbsp;·&nbsp; 📚&nbsp;<a href="#citation">Cite</a>
</p>
</div>

---

Language models hallucinate even as parameters, compute, and data scale. We show this is not only an
accuracy problem but a **second-moment reliability problem** — governed by the *covariance* between a
model's internal representations — and we give it a mechanism, a theory, and a demonstration.

**Neural diversity** — decorrelating the parallel representation streams inside a model — provably lowers
the tail probability of hallucination without adding parameters or data. A single scalar, the neural
diversity index $\mathcal{D}$, explains **94.3%** of the reliability variation across configurations
(Qwen2.5-0.5B, 20M Pile tokens, 12 tasks). As a demonstration, **ND-LoRA** (Neural Diversity Low-Rank
Adaptation) acts on it to reduce hallucinations by **up to 25.6% (14.6% on average)** while preserving
capability, at **+0.004%** pretraining cost and **1.1×** inference latency.

## Contributions

- **A reliability theory of hallucination.** We reframe hallucination as a second-moment problem and
  derive the *first formal tail bounds* on hallucination probability for ensembled language models,
  linking reliability directly to cross-stream representational covariance.
- **A measurable, causal mediator.** The neural diversity index $\mathcal{D}$ explains **94.3%** of
  reliability variation. Corruption interventions establish causality ($p < 0.001$); correlational
  analysis quantifies the slope (**+0.1%** neural correlation ↔ **+3.8%** hallucination).
- **ND-LoRA.** Parallel LoRA adapters + Barlow Twins decorrelation cut hallucinations by **up to 25.6%**
  (**14.6%** average, **+12.8%** at fixed $P{=}4$), synergistically — LoRA and regularization each help,
  and help more together.
- **Reliability without scale.** Gains arrive orthogonally to parameters and data, at near-zero cost
  (**+0.004%** pretraining, **1.1×** latency), and task-dependent optima emerge: different tasks want
  different amounts of diversity.

## Results

| Metric | Value |
|---|---|
| Hallucination reduction — best task-optimized $P$ | **25.6%** |
| Hallucination reduction — average | **14.6%** |
| Hallucination reduction — fixed $P{=}4$ | **12.8%** |
| Reliability variation explained by $\mathcal{D}$ | **94.3%** |
| Correlational slope (neural correlation → hallucination) | +0.1% → **+3.8%** |
| Causal corruption intervention | $p < 0.001$ |
| Added pretraining compute | **+0.004%** |
| Inference latency | **1.1×** |

*Setting: Qwen2.5-0.5B, 20M Pile tokens, 12 hallucination/faithfulness tasks (TruthfulQA, HaluEval,
MemoTrap, and others). See the [paper](https://openreview.net/pdf?id=5l9ZflyApA) for full tables and proofs.*

## The mechanism

A ParScale-style model runs $P$ parallel representation **streams** and aggregates them. When the streams
*collapse* — encoding the same thing — ensembling buys nothing and the model inherits the reliability of a
single stream. **Neural diversity** keeps the streams decorrelated so their errors average out.

We quantify collapse with the neural diversity index, defined on per-feature whitened stream
representations $\tilde{z}_i$:

```math
\mathcal{D} = \sqrt{\mathbb{E}_{i<j}\left[\frac{(\tilde{z}_i \cdot \tilde{z}_j)^2}{\|\tilde{z}_i\|^2 \|\tilde{z}_j\|^2}\right]}
```

where $\mathcal{D}=0$ means the streams are orthogonal (maximally diverse) and $\mathcal{D}=1$ means they
have fully collapsed.

ND-LoRA drives $\mathcal{D}$ down with two ingredients: **stream-specific LoRA adapters** (so streams can differ) and a **Barlow Twins decorrelation loss** (so they do). Hyperparameters (e.g. $\lambda_{\mathrm{BT}}$) are tuned with Optuna.

## Installation

Requires Python 3.10+ and a [Modal](https://modal.com) account for GPU execution.

```bash
git clone https://github.com/kushalc/nd-lora.git
cd nd-lora
uv sync            # uv-managed environment from pyproject.toml (single source of deps)
modal token new    # authenticate Modal once
```

## Quickstart

Every script launches its own Modal job programmatically (`app.run(detach=True)`, streamed logs) — invoke
them directly with `uv run`, not through `modal run`.

```bash
# Train an experiment from its YAML config (config path is positional)
uv run scripts/train_ndlora.py configs/ND-LoRA_P4.yaml

# Evaluate the paper checkpoints (mode = full | general | deep | quick)
uv run scripts/eval_experiments.py deep

# Causal corruption / dose-response analysis
uv run scripts/eval_neurodiversity.py --use-modal --meta-mode dose --corruption-mode stream --limit 128
```

## Reproducing the paper

Every experiment is one YAML in [`configs/`](configs/), named for its checkpoint. Training and evaluation
run entirely on Modal.

**Core results (Tables 1, 7–9)** — parameter-matched baselines and the ND-LoRA family:

```bash
uv run scripts/train_ndlora.py configs/Qwen2.5-0.5B_P1_R32.yaml   # P=1 baseline (rank sweep: R32/R64/R128)
uv run scripts/train_ndlora.py configs/ParScale_P4_R64.yaml       # ParScale baseline (P=2/4/8)
uv run scripts/train_ndlora.py configs/ND-LoRA_P4.yaml            # ND-LoRA, Optuna-optimized (P=2/4/8)
```

**Ablations (Tables 4, 6)** — component and module decompositions:

```bash
uv run scripts/train_ndlora.py configs/ParScale-BT_P4.yaml          # regularization only
uv run scripts/train_ndlora.py configs/Stream_LoRA_P4.yaml          # diversity capacity only
uv run scripts/train_ndlora.py configs/Stream_LoRA-BT_P4.yaml       # both (synergy)
uv run scripts/train_ndlora.py configs/ND-LoRA_P4_no_attention.yaml # module ablations (also _no_MLP)
```

**Evaluation and figures**:

```bash
uv run scripts/eval_experiments.py deep                                          # N=1024/task eval
uv run scripts/analyze_experiments.py --plot-mode all pub --analysis-mode full   # publication plots
```

## Model checkpoints

[`utils/model_checkpoints.py`](utils/model_checkpoints.py) is the single source of truth for every
paper-reproduction checkpoint, keyed by clean model name:

```python
from utils.model_checkpoints import CHECKPOINTS, DISPLAY_NAMES, MODEL_NAMES

CHECKPOINTS["ND-LoRA_P4"]     # -> loadable S3 checkpoint path
DISPLAY_NAMES["ND-LoRA_P4"]   # -> "ND-LoRA (P=4, OptC9)"
```

Covered: P=1 baselines (rank 32/64/128), ParScale (P=2/4/8), ND-LoRA (P=2/4/8), and the component/module
ablations.

## Repository structure

```
nd-lora/
├── scripts/                     # Modal entry points (train, evaluate, analyze)
│   ├── train_ndlora.py          #   YAML-driven training
│   ├── eval_experiments.py      #   hallucination benchmark evaluation
│   ├── eval_neurodiversity.py   #   causal corruption / dose-response analysis
│   └── analyze_experiments.py   #   results parsing + publication plots
├── configs/                     # one YAML per paper experiment (named for its checkpoint)
├── utils/model_checkpoints.py   # checkpoint registry (single source of truth)
├── adhoc/                       # figure/table generation
├── ParScale/                    # parallel-stream architecture (vendored)
├── leaderboard/                 # hallucination evaluation harness (vendored)
├── paper/                       # LaTeX source
└── pyproject.toml               # dependencies + packaging (single source)
```

## Citation

```bibtex
@article{chakrabarti2026neuraldiversity,
  title={Neural Diversity Regularizes Hallucinations in Language Models},
  author={Chakrabarti, Kushal and Balachundhar, Nirmal},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026},
  url={https://openreview.net/pdf?id=5l9ZflyApA}
}
```

## License & acknowledgments

MIT — see [LICENSE](LICENSE). Built on [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B),
[The Pile](https://pile.eleuther.ai/), the [ParScale](https://github.com/cli99/ParScale) architecture, and
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
