"""
Example Script to generate a vector index and upload it to wandb as an artifact

Things to change depending on your use case:
* wandb project and entity
* framework name
* retriever to use (CodeT5Retriever or SentenceTransformerRetriever)
* path to the documentation
* name of the artifact to be used while uploading the artifacts
"""

import weave
from dotenv import load_dotenv

import wandb
from ml_frameworks_bot.retrieval.codet5_retriever import CodeT5Retriever


load_dotenv()
wandb.init(
    project="ml-frameworks-bot",
    entity="ml-colabs",
    job_type="build_vector_index",
)
weave.init("ml-colabs/ml-frameworks-bot")
retriever = CodeT5Retriever(
    framework="keras3",
    repository_local_path="artifacts/keras3-docs:v0",
)
vector_index = retriever.index_documents(
    vector_index_persist_dir="artifacts/vector_indices/keras3_api_reference",
    artifact_name="keras3_api_reference",
)
