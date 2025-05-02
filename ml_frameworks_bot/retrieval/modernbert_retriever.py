import os
from typing import Optional

import safetensors
import torch
import torch.nn.functional as F
import weave
from rich.progress import track
from sentence_transformers import SentenceTransformer

import wandb

from ..utils import get_torch_backend, get_wandb_artifact, upload_as_artifact
from .common import FrameworkParams, argsort_scores, load_documents


class ModernBERTRetriever(weave.Model):
    framework: str
    embedding_model_name: str
    device: str
    repository_local_path: Optional[str] = None
    _model: Optional[SentenceTransformer] = None
    _vector_index: Optional[torch.Tensor] = None
    _documents: Optional[list[dict[str, str]]] = None

    def __init__(
        self,
        framework: str,
        embedding_model_name: str = "answerdotai/ModernBERT-base",
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
        self._model = SentenceTransformer(
            self.embedding_model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
            device=get_torch_backend(),
        )
        self._vector_index = vector_index
        self._documents = (
            load_documents(
                framework=framework, repository_local_path=repository_local_path
            )
            if documents is None
            else documents
        )

    def add_end_of_sequence_tokens(self, input_examples):
        input_examples = [
            input_example["text"]
            + self._model.tokenizer.added_tokens_decoder[50282].content
            # Source: https://huggingface.co/docs/transformers/main/en/model_doc/modernbert#transformers.ModernBertConfig.eos_token_id
            for input_example in input_examples
        ]
        return input_examples

    def index_documents(
        self,
        batch_size: int = 1,
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
                    embeddings = self._model.encode(
                        self.add_end_of_sequence_tokens(batch),
                        batch_size=len(batch),
                        normalize_embeddings=True,
                    )
                    vector_indices.append(torch.tensor(embeddings))

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
    ) -> "ModernBERTRetriever":
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
            query_embedding = torch.tensor(
                self._model.encode(
                    query,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
            ).cpu()
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
