"""
Example Script to generate a vector index and upload it to wandb as an artifact

Things to change depending on your use case:
* wandb project and entity
* framework name
* path to the documentation
* name of the artifact to be used while uploading the artifacts
"""

import weave
from dotenv import load_dotenv

import wandb
from ml_frameworks_bot.retrieval import NeuralRetreiver


load_dotenv()
wandb.init(
    project="ml-frameworks-bot",
    entity="ml-colabs",
    job_type="build_vector_index",
)
weave.init("ml-colabs/ml-frameworks-bot")
retriever = NeuralRetreiver(
    framework="mlx",
    embedding_model_name="BAAI/bge-small-en-v1.5",
    repository_local_path="artifacts/mlx-docs:v0",
)
vector_index = retriever.index_documents(
    vector_index_persist_dir="artifacts/vector_indices/mlx_api_reference",
    artifact_name="mlx_api_reference",
)
