#!/usr/bin/env python3
"""Generate LaTeX table from pub-full-single-stream.parquet"""

import argparse

import pandas as pd

# On-demand builder for the aggregated single-stream score parquet (from raw S3 evals).
from build_scores import ensure_scores_parquet

# Parse arguments
parser = argparse.ArgumentParser(description='Generate LaTeX benchmark table (Tables 7/8/9) from the single-stream score parquet')
parser.add_argument('--bold', choices=['none', 'within', 'global'], default='within',
                    help='Bold best scores: none, within (parameter-matched set), or global (across all models)')
parser.add_argument('--parquet', type=str, default=None,
                    help='Path to aggregated-results parquet (default: build on demand from raw S3 evals)')
args = parser.parse_args()

# Read data (regenerated from raw S3 evals if not provided)
df = pd.read_parquet(args.parquet or ensure_scores_parquet(plot_type="pub", mode="full"))

# Filter to Q0.5B models only (exclude Q1.5B and base models)
df_filtered = df[df.index.get_level_values(0) == 'Q0.5B'].copy()

# Define model selection and display names based on MODEL_CHECKPOINTS
# Map from simplified names in data to desired display names
SELECTED_MODELS = {
    # R32 models (P=2 equivalent)
    "Repro LoRA R32": "Qwen LoRA",
    "SharedLoRA R32": "ParScale",
    "nOSL IndLoRA": "ND-LoRA",  # Will appear in P=2, P=4, P=8
    "ND-LoRA [OptC9]": "ND-LoRA",  # Will appear in P=2, P=4, P=8

    # R64 models (P=4 equivalent)
    "Repro LoRA R64": "Qwen LoRA",
    "SharedLoRA R64": "ParScale",

    # R128 models (P=8 equivalent)
    "Repro LoRA R128": "Qwen LoRA",
    "SharedLoRA R128": "ParScale",
}

data = []
for idx, row in df_filtered.iterrows():
    model_size, p_val, model_name, full_name = idx
    if pd.isna(p_val):  # Skip NaN entries
        continue
    if model_name not in SELECTED_MODELS:  # Only include selected models
        continue

    # Map display name
    display_name = SELECTED_MODELS[model_name]

    data.append({
        'P': p_val,  # Keep as string like "P=1"
        'Model': display_name,
        'OriginalModel': model_name,  # Keep original for reference
        **{col: row[col] for col in df.columns}
    })

result_df = pd.DataFrame(data)

# Shorten column names for table
col_mapping = {
    'HaluEval Dialog, Accuracy': 'HE Dialog',
    'HaluEval QA, Accuracy': 'HE QA',
    'HaluEval Summarization, Accuracy': 'HE Summ',
    'MemoTrap, Accuracy': 'MemoTrap',
    'NQ (8-shot), EM': 'NQ-8',
    'TriviaQA (8-shot), EM': 'TQA-8',
    'TruthfulQA MC1, Accuracy': 'TF-MC1',
    'TruthfulQA MC2, Accuracy': 'TF-MC2',
    'nq swap, EM': 'NQ-swap',
    'popQA, EM': 'PopQA',
    "wikitext, bits_per_byte": "Wikitext BPB",
    "winogrande, Accuracy": "Winogrande",
}

# Rename columns for pivoting and drop columns not in col_mapping
result_df = result_df.rename(columns={k: v for k, v in col_mapping.items()})
# Keep only the columns we care about (Model, P, OriginalModel, and mapped eval columns)
eval_cols = [v for v in col_mapping.values()]
result_df = result_df[['Model', 'P', 'OriginalModel'] + eval_cols]

# Reassign P=1 models to correct P value based on rank before collision handling
def reassign_p_value(row):
    if row['P'] == 'P=1':
        orig_model = row['OriginalModel']
        if 'R128' in orig_model:
            return 'P=8'
        elif 'R64' in orig_model:
            return 'P=4'
        elif 'R32' in orig_model:
            return 'P=2'
    return row['P']

result_df['P'] = result_df.apply(reassign_p_value, axis=1)

# Handle collisions: when both nOSL IndLoRA and OptC9 map to ND-LoRA, keep only OptC9
# Group by (Model, P) and if there are duplicates, prefer OptC9
result_df['priority'] = result_df['OriginalModel'].apply(lambda x: 0 if 'OptC9' in x else 1)
result_df = result_df.sort_values('priority').groupby(['Model', 'P'], as_index=False).first()
result_df = result_df.drop(columns=['priority'])

# Group by rank mapping: R128 ~ P=8, R64 ~ P=4, R32 ~ P=2
# P values have already been reassigned, so just use them directly
rank_groups = {}
for _, row in result_df.iterrows():
    p_val = row['P']
    if p_val == 'P=8':
        group = 'R128 (P=8)'
    elif p_val == 'P=4':
        group = 'R64 (P=4)'
    elif p_val == 'P=2':
        group = 'R32 (P=2)'
    else:
        continue  # Skip unknown P values

    if group not in rank_groups:
        rank_groups[group] = []
    rank_groups[group].append(row)

# Convert to DataFrames
p_groups = {k: pd.DataFrame(v) for k, v in rank_groups.items()}

# Compute global best scores if needed
if args.bold == 'global':
    global_best = result_df[eval_cols].max()

# Generate subtables in order
subtables = ['2', '4', '8']
group_order = ['R32 (P=2)', 'R64 (P=4)', 'R128 (P=8)']
for (p_label, subtable_letter) in zip(group_order, subtables):
    p_df = p_groups.get(p_label)
    if p_df is None or p_df.empty:
        continue

    # Transpose: rows = evals, cols = models
    # Set Model as index, drop P and OriginalModel columns, then transpose
    pivot_df = p_df.drop(columns=['P', 'OriginalModel']).set_index('Model').T

    # Reorder columns: Qwen LoRA, ParScale, ND-LoRA
    desired_order = ['Qwen LoRA', 'ParScale', 'ND-LoRA']
    cols = [col for col in desired_order if col in pivot_df.columns]
    pivot_df = pivot_df[cols]

    print("\\begin{table}[htbp]")
    print("  \\centering")
    print("  \\begin{tabular}{l|" + "c" * len(pivot_df.columns) + "}")
    escaped_cols = [col.replace('_', '\\_') for col in pivot_df.columns]
    print("    \\textbf{Evaluation} & " + " & ".join([f"\\textbf{{{col}}}" for col in escaped_cols]) + " \\\\")
    print("    \\hline")

    for eval_name in pivot_df.index:
        values = []
        # Extract scalar values properly to avoid Series ambiguity
        row_vals = []
        for model_name in pivot_df.columns:
            val = pivot_df.loc[eval_name, model_name]
            # Handle case where duplicate column names return Series
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            row_vals.append(val)

        # For Wikitext BPB, lower is better; for others, higher is better
        is_lower_better = (eval_name == 'Wikitext BPB')

        # Determine which values to bold
        if args.bold == 'within':
            if is_lower_better:
                best_val = min(v for v in row_vals if not pd.isna(v)) if any(not pd.isna(v) for v in row_vals) else None
            else:
                best_val = max(v for v in row_vals if not pd.isna(v)) if any(not pd.isna(v) for v in row_vals) else None
        elif args.bold == 'global':
            if is_lower_better:
                best_val = result_df[eval_name].min() if eval_name in result_df.columns else None
            else:
                best_val = global_best.get(eval_name, None) if eval_name in global_best.index else None
        else:
            best_val = None

        for model_name in pivot_df.columns:
            val = pivot_df.loc[eval_name, model_name]
            if pd.isna(val):
                values.append("---")
            else:
                formatted = f"{val:.3f}"
                if best_val is not None and abs(val - best_val) < 1e-3:
                    formatted = f"\\textbf{{{formatted}}}"
                values.append(formatted)
        print(f"    {eval_name} & " + " & ".join(values) + " \\\\")

    print("  \\end{tabular}")
    print("  \\caption{")
    print(f"    Benchmark results for {p_label} parameter-matched models.")
    print("  }")
    print(f"  \\label{{tab:results_p{subtable_letter}}}")
    print("\\end{table}")
    print()
