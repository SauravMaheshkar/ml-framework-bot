import os
from typing import Optional

import safetensors
import torch
import torch.nn.functional as F
import weave
from rich.progress import track
from transformers import AutoModel, AutoTokenizer

import wandb

from ..utils import get_torch_backend, get_wandb_artifact, upload_as_artifact
from .common import FrameworkParams, argsort_scores, load_documents


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
            self.embedding_model_name,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self.embedding_model_name, trust_remote_code=True
        )
        self._vector_index = vector_index
        self._documents = (
            load_documents(
                framework=framework, repository_local_path=repository_local_path
            )
            if documents is None
            else documents
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
            self._model = self._model.to(self.device)

            with torch.no_grad():
                for idx in track(
                    range(0, len(self._documents), batch_size),
                    description=f"Encoding documents using {self.embedding_model_name}",
                ):
                    batch = self._documents[idx : idx + batch_size]
                    inputs = self._tokenizer(
                        [doc["text"] for doc in batch],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    embeddings = torch.unsqueeze(self._model(**inputs)[0], dim=0)
                    vector_indices.append(embeddings)

            vector_indices = torch.cat(vector_indices, dim=0).detach().cpu()

            if vector_index_persist_dir is not None:
                os.makedirs(vector_index_persist_dir, exist_ok=True)
                safetensors.torch.save_file(
                    {"vector_index": vector_indices},
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

            self._vector_index = vector_indices
            return vector_indices

    @classmethod
    def from_wandb_artifact(
        cls,
        artifact_address: str,
        repository_local_path: Optional[str] = None,
    ) -> "CodeT5Retriever":
        artifact_dir, metadata = get_wandb_artifact(
            artifact_name=artifact_address,
            artifact_type="vector_index",
            get_metadata=True,
        )
        with safetensors.torch.safe_open(
            os.path.join(artifact_dir, "vector_index.safetensors"), framework="pt"
        ) as f:
            vector_index = f.get_tensor("vector_index")
        return cls(
            framework=metadata.get("framework"),
            embedding_model_name=metadata.get("embedding_model_name"),
            vector_index=vector_index,
            documents=load_documents(
                framework=metadata.get("framework"),
                repository_local_path=repository_local_path,
            ),
        )

    @weave.op()
    def predict(self, query: str, top_k: int = 2) -> list[dict[str, str]]:
        with torch.no_grad():
            query_inputs = self._tokenizer(
                query,
                return_tensors="pt",
            )
            query_embedding = self._model(**query_inputs)[0]
            print(query_embedding.shape, self._vector_index.shape)
            scores = (
                F.cosine_similarity(query_embedding, self._vector_index)
                .cpu()
                .numpy()
                .tolist()
            )
            scores = argsort_scores(scores, descending=True)[:top_k]
        retrieved_chunks = []
        for score in scores[:top_k]:
            retrieved_chunks.append(
                {
                    **self._documents[score["original_index"]],
                    "score": score["item"],
                }
            )
        return retrieved_chunks
