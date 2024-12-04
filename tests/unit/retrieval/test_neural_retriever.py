import pytest
import weave
from dotenv import load_dotenv

from ml_frameworks_bot.retrieval import CodeT5Retriever


load_dotenv()


@pytest.mark.parametrize(
    "artifact_address, query, expected_path",
    [
        pytest.param(
            "ml-colabs/ml-frameworks-bot/keras3_api_reference:latest",
            "keras.layers.Dense",
            "sources/api/layers/core_layers/dense.md",
            id="keras3",
        ),
        pytest.param(
            "ml-colabs/ml-frameworks-bot/mlx_api_reference:latest",
            "mlx.nn.AvgPool1d",
            "python/nn/_autosummary/mlx.nn.AvgPool1d.txt",
            id="mlx",
        ),
    ],
)
def test_codet5_retriever(
    artifact_address: str, query: str, expected_path: str
) -> None:
    load_dotenv()
    weave.init(project_name="ml-colabs/ml-frameworks-bot")
    retriever = CodeT5Retriever.from_wandb_artifact(artifact_address=artifact_address)
    retrieved_nodes = retriever.predict(query=f"Fetch the API referece for `{query}`")
    assert retrieved_nodes[0]["file_path"] == expected_path
