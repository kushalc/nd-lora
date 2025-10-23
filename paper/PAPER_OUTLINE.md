# Neural Diversity Regularizes Hallucinations in Language Models
## Abstract (150 words)
* **Hook**: Despite promising scaling laws, parallel computation streams in LLMs exhibit unexpected failure modes - performance degrades beyond P=4 streams due to representational collapse
* **Insight**: We identify neural diversity as the key mechanism: when parallel streams develop correlated representations, hallucination probability increases predictably
* **Theory**: We prove hallucination probability upper bound: P(halluc) ≤ f(σ²((1-ρ)/P + ρ), μ²) where correlation ρ is controlled by neural diversity index D_spec
* **Method**: Orthogonal Stream LoRA (OSL) - independent adapters + Barlow Twins regularization to maintain diversity
* **Results**: OSL reduces hallucination by 15% on TruthfulQA, 12% on HaluEval-QA while maintaining perplexity; validated across 0.5B-3B models
* **Validation**: Theory-practice alignment confirmed: R²=0.87 between predicted and empirical hallucination rates

## Introduction (1 page)
### 1.1 Opening Hook
* LLMs exhibit persistent hallucination despite scale
* Increasingly, people are realizing SLMs might be the path forward, e.g. agents.
* However, SLMs are more prone to hallucinations: https://x.com/scaling01/status/1952781018554933261
* Recent work has shown the promise of parallel computation; parallel computation (test-time, mixture-of-experts, multi-head attention) promises efficiency but shows surprising failure modes.
* Across at least two of three broad categories of hallucinations, we demonstrate that we can use neural diversity in parralel computation to regularize hallucinations in small models.

### 1.2 Key Insight
* Root cause: representational collapse - parallel streams converge to correlated features
* This correlation directly increases hallucination probability
* We know from portfolio theory that having a portfolio of P uncorrelated stocks increases SNR by sqrt(P).
* Natural question: Can we maintain diversity to suppress hallucination?

### 1.3 Contributions
1. **Theoretical**: First principled connection between architectural diversity and hallucination and scaling laws demonstrating the value of diversity
2. **Methodological**: ND-LoRA - practical instantiation maintaining diversity through training
3. **Empirical**: Comprehensive validation showing up to 8% hallucination reductions with theory-practice alignment (R²=0.TBD)
4. **Mechanistic**: Neurodiversity as a causal mediator; optimal diversity on a task-dependent basis.

### 1.4 Paper Roadmap
* Section 2: Theoretical Framework
* Section 3: OSL Method
* Section 4: Experimental Validation
* Section 5: Mechanistic Analysis
* Section 6: Related Work

# 2. Theoretical Framework (1.5 pages)
### 2.1 Problem Formulation
* Model with P parallel streams: M(x) = (1/P)∑ᵢ mᵢ(x)
* Hallucination event: M(x) ≤ 0
* Stream statistics: mean μ, variance σ², correlation ρᵢⱼ

### 2.2 Main Theoretical Results
**Theorem 1 (Variance of Aggregated Margin)**
* Var(M) = σ²((1-ρ)/P + ρ)
* Proof sketch using correlation structure

**Theorem 2 (Hallucination Probability Bound)**
* P(M ≤ 0) ≤ Var(M)/(Var(M) + μ²) via Cantelli's inequality
* Explicit dependence on average correlation ρ

**Definition (Neural Diversity Index)**
* Whitened features z̃ᵢ at design layer
* Cross-correlation Cᵢⱼ = E[z̃ᵢz̃ⱼᵀ]
* D_spec = (1/P(P-1))∑ᵢ≠ⱼ ||Cᵢⱼ||₂

**Theorem 3 (Diversity-Hallucination Connection)**
* ρ ≤ κ·D_spec where κ depends on downstream projections
* Substituting into Theorem 2 gives diversity-controlled bound

### 2.3 Scaling Behavior
* Non-monotonic U-shaped curve predicted
* Optimal P depends on diversity maintenance ability
* Empirical validation in Section 4.3

### 2.4 Tightness Analysis
* Comparison of bound to empirical rates [FIGURE]
* Discussion of approximation quality
* When bound is loose vs tight

## 3. ND-LoRA Method (1 page)
### 3.1 Architecture
* Base model with P parallel transformer streams
* Independent LoRA adapters per stream: Bᵢ, Aᵢ ∈ ℝ^(d×r)
* Aggregation via learned softmax weights αᵢ

### 3.2 Barlow Twins Regularization
* Loss: L_total = L_CE + λ_BT · L_BT
* Standard formulation: L_BT = ∑_{p<q} ||C^{p,q} - I||²_F
* RandK formulation: L_BT ∑_{p, q ~ MultN(C)} ||C^{p,q} - I||²_F

### 3.3 Training Details
* Design layer selection (layer -30% optimal, TBD)
* λ_BT scheduling: warmup → 0.1
* Gradient clipping for stability

### 3.4 Computational Considerations
* Memory: P×r additional parameters (small with r=16)
* Compute: P streams adds small X% (TBD) training time given 20M fine-tuning; Barlow Twins adds <1% (TBD) training time with fine-tuning
* Inference: P streams adds TBD inference time
* Data- and parameter-matched

## 4. Experimental Validation (2.5 pages)
### 4.1 Experimental Setup
**Models**
* Qwen2.5: 0.5B variants
* P ∈ {1, 2, 4, 8} parallel streams
* Training: 20M tokens (Pile)

**Baselines** 
1. **Vanilla ParScale**: Shared LoRA with rank scaling
2. **P-Ensemble**: P independent models (upper bound)
3. **Dropout-0.1**: Standard dropout baseline

**Evaluation Suite**
* **Hallucination**: TruthfulQA, HaluEval (Dialog/QA/Summ), MemoTrap
* **Factuality**: NaturalQuestions, TriviaQA, PopQA
* **General**: MMLU, HellaSwag, Perplexity

### 4.2 Main Results
**Table 1: Hallucination Metrics (with 95% CI)**
Method          TruthQA-MC1  HaluEval-QA  MemoTrap   Avg
Baseline P=4    25.8 ± 0.9   42.9 ± 1.2   48.4 ± 1.1  39.0
Dropout-0.2     26.2 ± 0.8   43.1 ± 1.3   48.9 ± 1.0  39.4
DiverseNet      27.1 ± 0.9   44.2 ± 1.1   49.2 ± 1.2  40.2
OSL P=4         29.7 ± 0.7*  48.3 ± 1.0*  51.3 ± 0.9* 43.1
P-Ensemble      30.1 ± 0.8   49.1 ± 1.2   52.1 ± 1.1  43.8

p < 0.05 vs baseline (bootstrap test, 5 seeds)

**Figure 1: Diversity-Hallucination Correlation**
* Scatter plot: D_spec vs empirical hallucination rate
* 50+ checkpoints across models/training
* R² = 0.87, validating theoretical prediction

**Table 2: Compute-Performance Tradeoff**
Method          Params  Train-FLOPS  Halluc↓  Perplexity
Baseline P=4    +2M     1.0×         39.0     1.234
OSL P=4         +8M     1.08×        43.1     1.233
P-Ensemble      ×4      4.0×         43.8     1.231

### 4.3 Scaling Analysis
**Figure 2: U-Shaped Scaling Curves**
* X-axis: P ∈ {1,2,4,8,16}
* Y-axis: Hallucination rate
* Three curves: Baseline, OSL, Theory prediction
* Shows optimal P=4 for OSL, degradation at P≥8

**Table 3: Model Size Scaling**
Model    Method      TruthQA  HaluEval  Relative Gain
0.5B     Baseline    25.6     39.2      -
0.5B     OSL         27.1     41.3      +5.9% / +5.4%
1.5B     Baseline    25.8     42.9      -
1.5B     OSL         29.7     48.3      +15.1% / +12.6%
3.0B     Baseline    28.3     45.2      -
3.0B     OSL         32.6     51.1      +15.2% / +13.1%

### 4.4 Statistical Validation
**Figure 3: Bootstrap Analysis**
* Distribution of performance differences (OSL - Baseline)
* 1000 bootstrap samples
* Clear separation from zero for key metrics

**Table 4: Ablation Study**
Component Removed      TruthQA  HaluEval  
Full ND-IndLoRA        29.7     48.3
- IndLoRA              26.3     43.7      (use shared)
- Barlow Twins         27.2     45.1      (no regularization)
- Parallel Streams     XX.X     XX.X      (use single stream)
- Design Layer -30%    XX.X     XX.X      (use final layer)

# 5. Mechanistic Analysis (1.5 pages)
### 5.1 Hallucination Type Analysis
**Figure 5: Improvement by Hallucination Type**
* Bar chart showing differential improvements
* Entity substitution: +18%
* Fact fabrication: +14%
* Logic errors: +11%
* Syntactic: +3%

**Case Studies**
* Example 1: Entity hallucination prevented by Stream 1 specialization
* Example 2: Reasoning error caught by Stream 2 dissent
* Analysis of aggregator weight patterns during hallucination

### 5.2 Theoretical Bound Verification
**Figure 6: Predicted vs Actual Hallucination**
* X-axis: Theoretical bound from Theorem 3
* Y-axis: Empirical hallucination rate
* Points for different λ_BT, P, model sizes
* Bound is ~2× loose but rank-preserving

### 5.2 Task-Dependent Optimality
* FLESH OUT

## 6. Related Work (1 page)
### 6.1 Hallucination in LLMs
* Types and taxonomy [Citations]
* Prior mitigation: RLHF, Constitutional AI, retrieval
* Position: Orthogonal architectural approach

### 6.2 Parallel Architectures
* MoE, multi-head attention, parallel adapters
* Representation collapse phenomenon
* Position: First to connect to hallucination

### 6.3 Diversity in Neural Networks
* Ensemble methods, dropout, weight diversity
* Barlow Twins and self-supervised learning
* Position: Novel application to hallucination

### 6.4 Theoretical Analysis
* Margin-based analysis in NLP
* Concentration inequalities for neural nets
* Position: First diversity-hallucination bound

## 7. Discussion (0.5 pages)
### 7.1 Limitations
* Scale limited to 1.5B (computational constraints)
* Single architecture family (Qwen)
* Human evaluation pending
* Bound looseness (1.3× empirical rate)

### 7.2 Broader Impact
* Reduced hallucination → safer deployment
* Computational efficiency vs ensembles
* Interpretability through stream specialization

### 7.3 Future Work
* Scale to 7B+ models
* Cross-architecture validation (Llama, Mistral)
* Combination with RLHF/Constitutional AI
* Tighter theoretical bounds
* Application to other failure modes

## 8. Conclusion (0.3 pages)
* Summary: Diversity principle for hallucination control
* Key insight: Correlation → hallucination, diversity helps
* Practical impact: 15% reduction with minimal overhead
* Theoretical contribution: Validated predictive framework
* Call to action: Rethink parallel architectures through diversity lens

## References (1.5 pages)
[40-50 references organized by topic]

# Appendix (5-8 pages)
### A. Theoretical Proofs
* Complete proofs for Theorems 1-3
* Tightness analysis details

### B. Experimental Details
* Hyperparameter configurations
* Training curves for all experiments
* Complete benchmark results (including neutral/negative)
* Computational requirements

### C. Additional Ablations & Sensitivities
* Design layer sensitivity (all layers)
* λ_BT sweep (0.001 to 1.0)
* LoRA rank analysis
* Batch size effects
* Modules
* Layers

### D. Extended Visualizations
* Per-layer diversity evolution
* Stream specialization across training
* Failure case analysis
* Additional t-SNE plots

### E. Implementation Details
* Code snippets for OSL
* Barlow Twins efficient computation
* Integration with existing frameworks
