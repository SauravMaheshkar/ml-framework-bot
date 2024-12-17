import asyncio

import weave
from dotenv import load_dotenv

from ml_frameworks_bot.metrics import DocumentationAgentJudge
from ml_frameworks_bot.op_extraction import OpExtractor
from ml_frameworks_bot.retrieval import CodeT5Retriever, HeuristicRetreiver


load_dotenv()


def test_op_extractor_neural_retriever(keras3_docs):
    load_dotenv()
    weave.init(project_name="ml-colabs/ml-frameworks-bot")
    api_reference_retriever = CodeT5Retriever.from_wandb_artifact(
        artifact_address="ml-colabs/ml-frameworks-bot/keras3_api_reference:latest"
    )
    op_extractor = OpExtractor(
        model_name="claude-3-5-sonnet-20241022",
        api_reference_retriever=api_reference_retriever,
        verbose=False,
    )
    evaluation = weave.Evaluation(
        dataset=weave.ref("keras_evaluation_dataset:v2").get(),
        scorers=[
            DocumentationAgentJudge(
                repository_local_path=keras3_docs,
                column_map={
                    "ops": "keras_ops",
                    "api_reference_path": "keras_api_reference_path",
                },
            )
        ],
    )
    summary = asyncio.run(
        evaluation.evaluate(
            op_extractor, __weave={"display_name": "op_extractor_neural_retriever"}
        )
    )
    assert (
        summary["DocumentationAgentJudge"]["api_reference_retrieval_accuracy"]["mean"]
        > 0.6
    )
    assert summary["DocumentationAgentJudge"]["op_extraction_accuracy"]["mean"] > 0.6


def test_op_extractor_heuristic_retriever(keras3_docs):
    load_dotenv()
    weave.init(project_name="ml-colabs/ml-frameworks-bot")
    api_reference_retriever = HeuristicRetreiver(
        framework="keras3",
        repository_local_path=keras3_docs,
    )
    op_extractor = OpExtractor(
        model_name="claude-3-5-sonnet-20241022",
        api_reference_retriever=api_reference_retriever,
        verbose=False,
    )
    evaluation = weave.Evaluation(
        dataset=weave.ref("keras_evaluation_dataset:v2").get(),
        scorers=[
            DocumentationAgentJudge(
                repository_local_path=keras3_docs,
                column_map={
                    "ops": "keras_ops",
                    "api_reference_path": "keras_api_reference_path",
                },
            )
        ],
    )
    summary = asyncio.run(
        evaluation.evaluate(
            op_extractor, __weave={"display_name": "op_extractor_heuristic_retriever"}
        )
    )
    assert (
        summary["DocumentationAgentJudge"]["api_reference_retrieval_accuracy"]["mean"]
        == 1.0
    )
    assert summary["DocumentationAgentJudge"]["op_extraction_accuracy"]["mean"] > 0.8
