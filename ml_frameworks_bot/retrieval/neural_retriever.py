import json
import os
from typing import Optional

import safetensors
import torch
import torch.nn.functional as F
import wandb
import weave
from rich.progress import track
from sentence_transformers import SentenceTransformer

from ..utils import (
    get_all_file_paths,
    get_torch_backend,
    get_wandb_artifact,
    upload_file_as_artifact,
)
from .common import FrameworkParams, RepositoryMapping


def split_by_separator(
    split_pattern: list[tuple[str, str]], file_path: str, text: str
) -> list[dict[str, str]]:
    for pattern in split_pattern:
        if file_path.contains(pattern[0]):
            texts = text.split(pattern[1])
            return [{"file_path": file_path, "text": text} for text in texts]


def load_documents(
    self, repository_local_path: Optional[str] = None
) -> list[dict[str, str]]:
    if repository_local_path is None:
        repository_local_path = get_wandb_artifact(
            artifact_name=RepositoryMapping[self.framework]["artifact_address"],
            artifact_type="docs",
        )

    # Determine which files to index
    input_files = []
    for directory in FrameworkParams[self.framework]["included_directories"]:
        input_files += get_all_file_paths(
            directory=os.path.join(repository_local_path, directory),
            included_file_extensions=FrameworkParams[self.framework][
                "included_file_extensions"
            ],
        )

    documents = []
    for file_path in track(input_files, description="Loading documents"):
        # Exclude files with certain postfixes
        exclude_file = False
        if "exclude_file_postfixes" in FrameworkParams[self.framework]:
            for exclusion in FrameworkParams[self.framework]["exclude_file_postfixes"]:
                if file_path.endswith(exclusion):
                    exclude_file = True
                    break

        if not exclude_file:
            with open(file_path, "r") as file:
                text = file.read()

            if FrameworkParams[self.framework]["chunk_on_separator"]:
                documents.extend(
                    split_by_separator(
                        split_pattern=FrameworkParams[self.framework]["split_pattern"],
                        file_path=file_path,
                        text=text,
                    )
                )
            else:
                documents.append(
                    {
                        "file_path": file_path.replace(repository_local_path + "/", ""),
                        "text": text,
                    }
                )
    return documents


class NeuralRetreiver(weave.Model):
    framework: str
    embedding_model_name: str
    repository_local_path: Optional[str]
    _model: Optional[SentenceTransformer] = None
    _vector_index: Optional[torch.Tensor] = None
    _documents: Optional[list[dict[str, str]]] = None

    def __init__(
        self,
        framework: str,
        embedding_model_name: str,
        repository_local_path: Optional[str] = None,
        vector_index: Optional[torch.Tensor] = None,
        documents: Optional[list[dict[str, str]]] = None,
    ):
        super().__init__(framework=framework, embedding_model_name=embedding_model_name)
        self.repository_local_path = repository_local_path
        self._model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
            device=get_torch_backend(),
        )
        self._vector_index = vector_index
        self._documents = documents or load_documents(
            repository_local_path=repository_local_path
        )

    def add_end_of_sequence_tokens(self, input_examples):
        input_examples = [
            input_example + self._model.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples

    def index_documents(
        self,
        batch_size: int = 8,
        normalize_embeddings: bool = True,
        vector_index_persist_dir: Optional[str] = None,
        artifact_name: Optional[str] = None,
        artifact_aliases: Optional[list[str]] = [],
    ) -> torch.Tensor:
        if self.repository_local_path is not None:
            vector_indices = []
            for idx in track(
                range(0, len(self._documents), batch_size),
                description=f"Encoding documents using {self.embedding_model_name}",
            ):
                batch = self._documents[idx : idx + batch_size]
                batch_embeddings = self._model.encode(
                    self.add_end_of_sequence_tokens(batch),
                    batch_size=len(batch),
                    normalize_embeddings=normalize_embeddings,
                )
                vector_indices.append(torch.tensor(batch_embeddings))

            self._vector_index = torch.cat(vector_indices, dim=0)
            self._vector_index = self._vector_index.detach().cpu()

            if vector_index_persist_dir is not None:
                safetensors.torch.save_file(
                    {"vector_index": self._vector_index.cpu()},
                    os.path.join(vector_index_persist_dir, "vector_index.safetensors"),
                )
                assert (
                    wandb.run is not None
                ), "Attempted to log artifact without wandb run"
                upload_file_as_artifact(
                    path=os.path.join(
                        vector_index_persist_dir, "vector_index.safetensors"
                    ),
                    artifact_name=artifact_name,
                    artifact_metadata={
                        "framework": self.framework,
                        "embedding_model_name": self.embedding_model_name,
                        "torch_dtype": self.torch_dtype,
                        **{
                            key: FrameworkParams[self.framework][key]
                            for key in FrameworkParams[self.framework]
                        },
                    },
                    artifact_aliases=artifact_aliases,
                )

            return self._vector_index

    @classmethod
    def from_wandb_artifact(
        cls,
        artifact_address: str,
        repository_local_path: Optional[str] = None,
    ) -> "NeuralRetreiver":
        artifact_dir, metadata = get_wandb_artifact(
            artifact_name=artifact_address,
            artifact_type="vector_index",
            get_metadata=True,
        )
        with safetensors.torch.safe_open(
            os.path.join(artifact_dir, "vector_index.safetensors"), framework="pt"
        ) as f:
            vector_index = f.get("vector_index")
        device = torch.device(get_torch_backend())
        vector_index = vector_index.to(device)
        with open(os.path.join(artifact_dir, "config.json"), "r") as config_file:
            metadata = json.load(config_file)
        return cls(
            framework=metadata.get("framework"),
            embedding_model_name=metadata.get("embedding_model_name"),
            vector_index=vector_index,
            documents=load_documents(repository_local_path=repository_local_path),
        )

    @weave.op()
    def predict(self, query: list[str], top_k: int = 2) -> list[dict[str, str]]:
        device = torch.device(get_torch_backend())
        with torch.no_grad():
            query_embedding = self._model.encode(
                self.add_end_of_sequence_tokens(query),
                normalize_embeddings=True,
            )
            query_embedding = torch.from_numpy(query_embedding).to(device)
            scores = (
                F.cosine_similarity(query_embedding, self._vector_index)
                .cpu()
                .numpy()
                .tolist()
            )
        retrieved_chunks = []
        for score in scores:
            retrieved_chunks.append(
                {
                    **self._documents[score["original_index"]],
                    "score": score["item"],
                }
            )
        return retrieved_chunks
