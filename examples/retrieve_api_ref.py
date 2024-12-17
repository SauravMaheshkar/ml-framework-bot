"""
Example Script to generate a vector index and upload it to wandb as an artifact

Things to change depending on your use case:
* weave project where you to want to log the trace
* retriever to use (CodeT5Retriever or HeuristicRetriever)
* vector index you want to use
* the query you want to retrieve
"""

import weave
from dotenv import load_dotenv

from ml_frameworks_bot.retrieval import CodeT5Retriever


load_dotenv()
weave.init(project_name="ml-colabs/ml-frameworks-bot")
retriever = CodeT5Retriever.from_wandb_artifact(
    artifact_address="ml-colabs/ml-frameworks-bot/keras3_api_reference:latest"
)
retrieved_nodes = retriever.predict(
    query="Fetch the API referece for `keras.layers.Dense`"
)
