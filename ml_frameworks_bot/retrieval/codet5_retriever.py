import os
from typing import Optional

import safetensors
import torch
import weave
from rich.progress import track
from transformers import AutoModel, AutoTokenizer

import wandb

from ..utils import get_torch_backend, upload_as_artifact
from .common import FrameworkParams, load_documents


class CodeT5Retriever(weave.Model):
    framework: str
    embedding_model_name: str
    device: str
    repository_local_path: Optional[str] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _model: Optional[AutoModel] = None
    _vector_index: Optional[torch.Tensor] = None
    _documents: Optional[list[dict[str, str]]] = None

    def __init__(
        self,
        framework: str,
        embedding_model_name: str = "Salesforce/codet5p-110m-embedding",
        repository_local_path: Optional[str] = None,
        vector_index: Optional[torch.Tensor] = None,
        documents: Optional[list[dict[str, str]]] = None,
    ):
        super().__init__(
            framework=framework,
            embedding_model_name=embedding_model_name,
            device=get_torch_backend(),
        )
        self.repository_local_path = repository_local_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.embedding_model_name, trust_remote_code=True
        )
        self._vector_index = vector_index
        self._documents = documents or load_documents(
            framework=framework, repository_local_path=repository_local_path
        )

    def index_documents(
        self,
        batch_size: int = 8,
        vector_index_persist_dir: Optional[str] = None,
        artifact_name: Optional[str] = None,
        artifact_aliases: Optional[list[str]] = [],
    ) -> torch.Tensor:
        if self.repository_local_path is not None:
            vector_indices = []
            with torch.no_grad():
                for idx in track(
                    range(0, len(self._documents), batch_size),
                    description=f"Encoding documents using {self.embedding_model_name}",
                ):
                    batch = self._documents[idx : idx + batch_size]
                    inputs = self._tokenizer(
                        batch, padding=True, truncation=True, return_tensors="pt"
                    ).to(self.device)
                    vector_indices.append(self._model(inputs)[0])
            vector_indices = torch.cat(vector_indices, dim=0).detach().cpu()

            if vector_index_persist_dir is not None:
                os.makedirs(vector_index_persist_dir, exist_ok=True)
                safetensors.torch.save_file(
                    {"vector_index": self._vector_index.cpu()},
                    os.path.join(vector_index_persist_dir, "vector_index.safetensors"),
                )
                assert (
                    wandb.run is not None
                ), "Attempted to log artifact without wandb run"
                upload_as_artifact(
                    path=os.path.join(
                        vector_index_persist_dir, "vector_index.safetensors"
                    ),
                    artifact_name=artifact_name,
                    artifact_metadata={
                        "framework": self.framework,
                        "embedding_model_name": self.embedding_model_name,
                        **{
                            key: FrameworkParams[self.framework][key]
                            for key in FrameworkParams[self.framework]
                        },
                    },
                    artifact_aliases=artifact_aliases,
                )

            return self._vector_index
