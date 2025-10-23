#!/usr/bin/env python3

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from src.backend.envs import EVAL_REQUESTS_PATH_BACKEND, EVAL_RESULTS_PATH_BACKEND
from src.envs import QUEUE_REPO, RESULTS_REPO
from src.utils import my_snapshot_download
from utils.model_checkpoints_paper import ALL_CHECKPOINTS as MODEL_CHECKPOINTS, MODEL_NAMES, MODEL_SPACERS

logging.basicConfig(format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

EVAL_BLACKLIST = [
    ("stderr", None),
    ("f1", None),
    (None, "faithdial"),
    (None, "truthfulqa_gen"),
    (None, "fever"),
]
MODEL_WHITELIST = [
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-7b-hf",
    "ParControl/",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
]


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def find_json_files(json_path):
    res = []
    for root, dirs, files in os.walk(json_path):
        for file in files:
            if file.endswith(".json"):
                res.append(os.path.join(root, file))
    return res


def sanitise_metric(name: str) -> str:
    res = name
    res = res.replace("prompt_level_strict_acc", "Prompt-Level Accuracy")
    res = res.replace("acc", "Accuracy")
    res = res.replace("exact_match", "EM")
    res = res.replace("avg-selfcheckgpt", "AVG")
    res = res.replace("max-selfcheckgpt", "MAX")
    res = res.replace("rouge", "ROUGE-")
    res = res.replace("bertscore_precision", "BERT-P")
    res = res.replace("exact", "EM")
    res = res.replace("HasAns_EM", "HasAns")
    res = res.replace("NoAns_EM", "NoAns")
    res = res.replace("em", "EM")
    return res


def sanitise_dataset(name: str) -> str:
    res = name
    res = res.replace("tqa8", "TriviaQA (8-shot)")
    res = res.replace("nq8", "NQ (8-shot)")
    res = res.replace("nq_open", "NQ (64-shot)")
    res = res.replace("triviaqa", "TriviaQA (64-shot)")
    res = res.replace("truthfulqa", "TruthfulQA")
    res = res.replace("ifeval", "IFEval")
    res = res.replace("selfcheckgpt", "SelfCheckGPT")
    res = res.replace("truefalse_cieacf", "True-False")
    res = res.replace("mc", "MC")
    res = res.replace("race", "RACE")
    res = res.replace("squad", "SQuAD")
    res = res.replace("memo-trap", "MemoTrap")
    res = res.replace("cnndm", "CNN/DM")
    res = res.replace("xsum", "XSum")
    res = res.replace("qa", "QA")
    res = res.replace("summarization", "Summarization")
    res = res.replace("dialogue", "Dialog")
    res = res.replace("halueval", "HaluEval")
    res = res.replace("_v2", "")
    res = res.replace("_", " ")
    return res


def sanitise_model(name: str) -> str:
    return MODEL_NAMES.get(name, name)


def generate_plots(plot_mode=True, spacer_after=None):
    """Generate plots in either clustermap or standard sorted mode.

    Args:
        clustermap_mode: Whether to use hierarchical clustering
        spacer_after: List of column names after which to insert spacer columns
    """
    # Try to load the data_map from the cache file
    my_snapshot_download(repo_id=RESULTS_REPO, revision="main", local_dir=EVAL_RESULTS_PATH_BACKEND, repo_type="dataset", max_workers=60)
    my_snapshot_download(repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60)

    result_path_lst = find_json_files(EVAL_RESULTS_PATH_BACKEND)
    model_dataset_metric_to_result_map = {}
    data_map = {}
    for path in result_path_lst:
        to_add = False
        for name in MODEL_WHITELIST:
            if name in path:
                to_add = True
                break
        if not to_add:
            continue

        try:
            with open(path, 'r') as f:
                data = json.load(f)
            model_name = data.get("config", {}).get("model_name")
            if not model_name:
                logging.debug("Skipped %s without model_name", path)
                continue

            for dataset_name, results_dict in data["results"].items():
                for metric_name, value in results_dict.items():
                    to_add = True
                    for metric_regex, data_regex in EVAL_BLACKLIST:
                        if metric_regex is not None and metric_regex in metric_name:
                            to_add = False
                            break
                        elif data_regex is not None and data_regex in dataset_name:
                            to_add = False
                            break

                    if 'bertscore' in metric_name:
                        if 'precision' not in metric_name:
                            to_add = False

                    if 'halueval' in dataset_name:
                        if 'acc' not in metric_name:
                            to_add = False

                    if 'ifeval' in dataset_name:
                        if 'prompt_level_strict_acc' not in metric_name:
                            to_add = False

                    if 'squad' in dataset_name:
                        # to_add = False
                        if 'best_exact' in metric_name:
                            to_add = False

                    if "truthfulqa_gen" in dataset_name:
                        if "acc" not in metric_name:
                            to_add = False
                        if "rouge" in metric_name:
                            to_add = False

                    if ('xsum' in dataset_name or 'cnn' in dataset_name) and 'v2' not in dataset_name:
                        to_add = False

                    if isinstance(value, str):
                        if is_float(value):
                            value = float(value)
                        else:
                            to_add = False

                    if to_add:
                        if 'rouge' in metric_name:
                            value /= 100.0

                        if 'squad' in dataset_name:
                            value /= 100.0

                        sanitised_metric_name = metric_name
                        if "," in sanitised_metric_name:
                            sanitised_metric_name = sanitised_metric_name.split(',')[0]
                        sanitised_metric_name = sanitise_metric(sanitised_metric_name)
                        sanitised_dataset_name = sanitise_dataset(dataset_name)
                        model_name = sanitise_model(model_name)

                        if model_name not in data_map:
                            data_map[model_name] = {}
                        subkey = (sanitised_dataset_name, sanitised_metric_name)
                        key = (model_name,) + subkey
                        if np.abs(model_dataset_metric_to_result_map.get(key, value) - value) > 1e-3 or \
                           np.abs(data_map[model_name].get(subkey, value) - value) > 1e-3:
                            result = min(model_dataset_metric_to_result_map.get(key, value), value)
                            logging.warning("Chose minimum value for conflicted key=%s: %.3f %.3f -> %.3f",
                                            key, value, model_dataset_metric_to_result_map[key], result)
                            model_dataset_metric_to_result_map[key] = result
                            data_map[model_name][subkey] = result
                        else:
                            model_dataset_metric_to_result_map[key] = value
                            data_map[model_name][subkey] = value

                        # print('model_name', model_name, 'dataset_name', sanitised_dataset_name, 'metric_name', sanitised_metric_name, 'value', value)
        except:
            logging.error("Couldn't parse %s", path, exc_info=True)

    for plot_type in ['all', 'summ', 'qa', 'instr', 'detect', 'rc']:
        data_map_v2 = {}
        for model_name in data_map.keys():
            for dataset_metric in data_map[model_name].keys():
                if dataset_metric not in data_map_v2:
                    data_map_v2[dataset_metric] = {}

                if plot_type in {'all'}:
                    data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
                elif plot_type in {'summ'}:
                    if 'CNN' in dataset_metric[0] or 'XSum' in dataset_metric[0]:
                        data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
                elif plot_type in {'qa'}:
                    if 'TriviaQA' in dataset_metric[0] or 'NQ' in dataset_metric[0] or 'TruthfulQA' in dataset_metric[0]:
                        data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
                elif plot_type in {'instr'}:
                    if 'MemoTrap' in dataset_metric[0] or 'IFEval' in dataset_metric[0]:
                        data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
                elif plot_type in {'detect'}:
                    if 'HaluEval' in dataset_metric[0] or 'SelfCheck' in dataset_metric[0]:
                        data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
                elif plot_type in {'rc'}:
                    if 'RACE' in dataset_metric[0] or 'SQuAD' in dataset_metric[0]:
                        data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
                else:
                    assert False, f"Unknown plot type: {plot_type}"

        # df = pd.DataFrame.from_dict(data_map, orient='index')   # Invert the y-axis (rows)
        df = pd.DataFrame.from_dict(data_map_v2, orient='index')   # Invert the y-axis (rows)
        df.index = [', '.join(map(str, idx)) for idx in df.index]

        counts_s = df.count(axis=1)
        df = df.reindex(counts_s[counts_s >= 0.5 * counts_s.max()].index)

        # Sort columns and rows lexically if not in clustermap mode
        df = df.sort_index(axis=0, key=lambda x: x.str.lower())  # Sort rows (evaluations) lexically
        df = df.reindex(sorted(df.columns), axis=1)  # Sort columns (models) lexically

        # Insert spacer columns if requested
        if spacer_after and plot_mode != "clustermap":
            new_columns = []
            for col in df.columns:
                new_columns.append(col)
                if any(pat in col for pat in spacer_after):
                    name = f"__spacer__{len(new_columns)}"
                    new_columns.append(name)
                    df[name] = np.nan  # Add empty spacer column

            # Reorder columns to include spacers
            df = df[new_columns]

        # scaler = MinMaxScaler()
        # df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        df[df < 1e-3] = np.nan

        # Calculate dimensions based on the DataFrame size
        cell_height = 1  # Height of each cell in inches
        cell_width = 1   # Width of each cell in inches

        n_rows = len(df.index)  # Datasets and Metrics
        n_cols = len(df.columns)  # Models

        # Calculate figure size dynamically
        fig_width = cell_width * n_cols + 8  # colorbars, etc.
        fig_height = cell_height * n_rows + 12  # axis labels

        sns.set_context("notebook", font_scale=1.3)
        df.to_csv(f'plots/{plot_type}.tsv', float_format="%.3f", sep="\t")
        df.to_parquet(f'plots/{plot_type}.parquet')

        for cmap in [
            # None,
            # 'coolwarm',
            'viridis'
        ]:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), tight_layout=True)
            kwargs = dict(
                cmap=cmap,
                fmt='.1%',
                annot_kws={"size": 12},
                square=True,
                annot=True,
                cbar_kws={"fraction": .07, "shrink": 0.8},
                # linewidths=0.5,
            )
            if plot_mode == "clustermap":
                dendrogram_ratio = (.1, .1)
                fig = sns.clustermap(df, method='ward', metric='euclidean', dendrogram_ratio=dendrogram_ratio,
                                     ax=ax, **kwargs)
            else:
                sns.heatmap(df, ax=ax, **kwargs)
                ax.set_xticklabels([c if not c.startswith("__spacer__") else "" for c in df.columns])

            plt.xticks(rotation=90)
            plt.yticks(rotation=0)

            cmap_suffix = '' if cmap is None else f'_{cmap}'
            filename = f'plots/{plot_type}_{cmap_suffix}.png'
            fig.savefig(filename)

            logging.info(f"Generated plot @ {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate analysis plots for hallucination leaderboard")
    parser.add_argument("--plot-mode", choices=["clustermap", "sorted"], default="sorted",
                        help="Display mode: 'clustermap' for hierarchical clustering, 'sorted' for lexical sorting")
    parser.add_argument("--spacer-after", nargs="+", default=MODEL_SPACERS,
                        help="Add spacer columns after columns containing these patterns (e.g., 'ParControl/P=2/2025-09-14-23-00-29')")

    args = parser.parse_args()
    os.makedirs('plots', exist_ok=True)
    generate_plots(**vars(args))


if __name__ == "__main__":
    main()
