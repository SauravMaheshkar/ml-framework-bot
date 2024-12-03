import os
from typing import Any, Dict, Optional

from rich.progress import track

from ..utils import get_all_file_paths, get_wandb_artifact


RepositoryMapping: Dict[str, Dict[str, str]] = {
    "keras3": {
        "repository": "https://github.com/keras-team/keras-io",
        "artifact_address": "ml-colabs/ml-frameworks-bot/keras3-docs:latest",
        "mapping": "mappings/keras3.json",
    },
    "keras2": {
        "repository": "https://github.com/keras-team/keras-io",
        "artifact_address": "ml-colabs/ml-frameworks-bot/keras2-docs:latest",
    },
    "mlx": {
        "repository": "https://github.com/ml-explore/mlx",
        "artifact_address": "ml-colabs/ml-frameworks-bot/mlx-docs:latest",
        "mapping": "mappings/mlx.json",
    },
    "jax": {
        "repository": "https://github.com/jax-ml/jax",
        "artifact_address": "ml-colabs/ml-frameworks-bot/jax-docs:latest",
    },
    "flax": {
        "repository": "https://github.com/google/flax",
        "artifact_address": "ml-colabs/ml-frameworks-bot/flax-nnx-docs:latest",
    },
    "numpy": {
        "repository": "https://github.com/numpy/numpy",
        "artifact_address": "ml-colabs/ml-frameworks-bot/numpy-docs:latest",
    },
}

FrameworkParams: Dict[str, Dict[str, Any]] = {
    "keras3": {
        "chunk_on_separator": True,
        "split_pattern": [("ops", "----")],
        "included_directories": ["sources/api"],
        "included_file_extensions": [".md"],
    },
    "mlx": {
        "chunk_on_separator": False,
        "included_directories": ["dev", "examples", "python", "usage"],
        "included_file_extensions": [".txt"],
    },
}


def load_documents(
    framework: str, repository_local_path: Optional[str] = None
) -> list[dict[str, str]]:
    if repository_local_path is None:
        repository_local_path = get_wandb_artifact(
            artifact_name=RepositoryMapping[framework]["artifact_address"],
            artifact_type="docs",
        )

    # Determine which files to index
    input_files = []
    for directory in FrameworkParams[framework]["included_directories"]:
        input_files += get_all_file_paths(
            directory=os.path.join(repository_local_path, directory),
            included_file_extensions=FrameworkParams[framework][
                "included_file_extensions"
            ],
        )

    documents = []
    for file_path in track(input_files, description="Loading documents"):
        # Exclude files with certain postfixes
        exclude_file = False
        if "exclude_file_postfixes" in FrameworkParams[framework]:
            for exclusion in FrameworkParams[framework]["exclude_file_postfixes"]:
                if file_path.endswith(exclusion):
                    exclude_file = True
                    break

        if not exclude_file:
            with open(file_path, "r") as file:
                text = file.read()

            if FrameworkParams[framework]["chunk_on_separator"]:
                for pattern in FrameworkParams[framework]["split_pattern"]:
                    if pattern[0] in file_path:
                        texts = text.split(pattern[1])
                        element = [
                            {"file_path": file_path, "text": text} for text in texts
                        ]
                        documents.extend(element)
            documents.append(
                {
                    "file_path": file_path.replace(repository_local_path + "/", ""),
                    "text": text,
                }
            )

    return documents
