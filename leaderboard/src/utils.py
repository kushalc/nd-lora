import pandas as pd
from huggingface_hub import snapshot_download

from src.backend.envs import Task, Tasks


def my_snapshot_download(repo_id, revision, local_dir, repo_type, max_workers):
    for i in range(10):
        try:
            snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir, repo_type=repo_type, max_workers=max_workers)
            return
        except Exception:
            import time
            time.sleep(60)
    return


def get_dataset_url(row):
    dataset_name = row['Benchmark']
    dataset_url = row['Dataset Link']
    benchmark = f'<a target="_blank" href="{dataset_url}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{dataset_name}</a>'
    return benchmark


def get_dataset_summary_table(file_path):
    df = pd.read_csv(file_path)

    df['Benchmark'] = df.apply(lambda x: get_dataset_url(x), axis=1)

    df = df[['Category', 'Benchmark', 'Data Split', 'Data Size', 'Language']]

    return df


def get_tasks_by_benchmarks(benchmark_names: list[str] = None) -> list[Task]:
    """
    Helper function to filter pending tasks by the desired benchmark.
    """
    if benchmark_names is None:
        return [task.value for task in Tasks]
    
    # Get all available benchmark names for validation
    available_benchmarks = {task.value.benchmark for task in Tasks}
    
    invalid_benchmarks = set(benchmark_names) - available_benchmarks
    if invalid_benchmarks:
        raise ValueError(f"Invalid benchmark names: {invalid_benchmarks}. "
                        f"Available benchmarks: {sorted(available_benchmarks)}")
    
    return [task.value for task in Tasks if task.value.benchmark in benchmark_names]