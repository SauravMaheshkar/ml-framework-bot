import pytest
from dotenv import load_dotenv

from ml_frameworks_bot.retrieval import HeuristicRetreiver


load_dotenv()


@pytest.mark.parametrize(
    "framework, repository_local_path, query, expected_path",
    [
        (
            "keras3",
            "keras3_docs",
            "keras.layers.Dense",
            "sources/api/layers/core_layers/dense.md",
        ),
        ("mlx", "mlx_docs", "mlx.nn.CELU", "python/nn/_autosummary/mlx.nn.CELU.txt"),
    ],
    ids=["keras3", "mlx"],
)
def test_keras_retriever(
    framework, repository_local_path, query, expected_path, request
):
    retriever = HeuristicRetreiver(
        framework=framework,
        repository_local_path=request.getfixturevalue(repository_local_path),
    )
    retrieved_nodes = retriever.predict(query=query)

    assert retrieved_nodes["file_path"] == expected_path
