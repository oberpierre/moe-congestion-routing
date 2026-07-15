from pathlib import PurePosixPath


def available_clusters(dataset_repo: str) -> list[str]:
    """Discover cluster folder names (parents of parquet files) in the HF dataset repo."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(dataset_repo, repo_type="dataset")
    clusters = {
        PurePosixPath(f).parts[-2]
        for f in files
        if f.endswith(".parquet") and len(PurePosixPath(f).parts) >= 2
    }
    return sorted(clusters)
