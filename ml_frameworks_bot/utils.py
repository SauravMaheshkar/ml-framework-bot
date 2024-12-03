import json
import os
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
import weave
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel

import wandb


class SupportedFrameworks(BaseModel):
    frameworks: Literal["keras3", "keras2", "pytorch", "mlx", "flax", "jax", "numpy"]


def get_all_file_paths(
    directory: str, included_file_extensions: List[str]
) -> List[str]:
    file_paths = []

    def recurse_folder(folder):
        for entry in os.scandir(folder):
            if entry.is_dir():
                recurse_folder(entry.path)
            else:
                for file_ext in included_file_extensions:
                    if entry.path.endswith(file_ext):
                        file_paths.append(entry.path)

    recurse_folder(directory)

    return file_paths


def make_embedding_model(
    embedding_model_name: str, model_kwargs: Optional[Dict[str, Any]] = {}
) -> BaseEmbedding:
    return (
        OpenAIEmbedding(model_name=embedding_model_name)
        if embedding_model_name
        in [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]
        else HuggingFaceEmbedding(
            model_name=embedding_model_name,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
    )


def weave_op_wrapper(name: str) -> Callable[[Callable], Callable]:
    def wrapper(fn: Callable) -> Callable:
        op = weave.op()(fn)
        op.name = name
        return op

    return wrapper


def get_wandb_artifact(
    artifact_name: str,
    artifact_type: str,
    get_metadata: bool = False,
) -> str:
    if wandb.run:
        artifact = wandb.use_artifact(artifact_name, type=artifact_type)
        artifact_dir = artifact.download()
    else:
        api = wandb.Api()
        artifact = api.artifact(artifact_name)
        artifact_dir = artifact.download()
    if get_metadata:
        return artifact_dir, artifact.metadata
    return artifact_dir


def upload_file_as_artifact(
    path: str,
    artifact_name: str,
    artifact_metadata: Optional[Dict[str, Any]] = {},
    artifact_aliases: Optional[List[str]] = [],
) -> None:
    if wandb.run and artifact_name:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="vector_index",
            metadata=artifact_metadata,
        )
        if os.path.isdir(path):
            artifact.add_dir(local_path=path)
        elif os.path.isfile(path):
            artifact.add_file(local_path=path)
        else:
            raise ValueError(f"Path {path} is not a valid file or directory")
        wandb.log_artifact(artifact, aliases=artifact_aliases)


def get_structured_output_from_completion(
    completion: Any, response_format: type
) -> Any:
    return response_format.model_validate(
        json.loads(completion.choices[0].message.content)
    )


def get_torch_backend():
    if torch.cuda.is_available():
        if torch.backends.cuda.is_built():
            return "cuda"
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return "mps"
        return "cpu"
    return "cpu"
