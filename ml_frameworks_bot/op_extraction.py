from typing import Dict, List, Optional, Union

import litellm
import weave
from pydantic import BaseModel
from rich.progress import track

from .prompts import OP_EXTRACTION_TMPL
from .retrieval import CodeT5Retriever, HeuristicRetreiver
from .schema import Operations
from .utils import get_structured_output_from_completion


NeuralRetriever = Union[CodeT5Retriever]
DocumentationRetreiver = Union[NeuralRetriever, HeuristicRetreiver]


class OpWithAPIReference(BaseModel):
    op: str
    api_reference: str
    api_reference_path: str


class OpExtractor(weave.Model):
    model_name: str
    api_reference_retriever: DocumentationRetreiver
    verbose: bool

    def __init__(
        self,
        model_name: str,
        api_reference_retriever: DocumentationRetreiver,
        verbose: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            api_reference_retriever=api_reference_retriever,
            verbose=verbose,
        )

    @weave.op()
    def extract_operations(
        self, code_snippet: str, seed: Optional[int] = None, max_retries: int = 3
    ) -> Operations:
        completion = litellm.completion(
            model=self.model_name,
            response_format=Operations,
            seed=seed,
            messages=[
                {
                    "role": "system",
                    "content": OP_EXTRACTION_TMPL.format(
                        framework="keras"
                        if self.api_reference_retriever.framework
                        in ["keras3", "keras2"]
                        else self.api_reference_retriever.framework
                    ),
                },
                {
                    "role": "user",
                    "content": code_snippet,
                },
            ],
        )
        operations = get_structured_output_from_completion(
            completion=completion, response_format=Operations
        )
        unique_ops = Operations(operations=list(set(operations.operations)))
        return unique_ops

    @weave.op()
    def ask_llm_about_op(self, op: str) -> str:
        return litellm.completion(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": f"Describe the purpose of `{op}` in less than 100 words",  # noqa: E501
                }
            ],
        )

    @weave.op()
    def retrieve_api_references(self, ops: Operations) -> List[OpWithAPIReference]:
        ops_with_api_reference = []
        iterable = ops.operations
        iterable = (
            track(iterable, description="Retrieving api references:")
            if self.verbose
            else iterable
        )
        for op in iterable:
            if isinstance(self.api_reference_retriever, NeuralRetriever):
                purpose_of_op = self.ask_llm_about_op(op)
                api_reference = self.api_reference_retriever.predict(
                    query=f"API reference for `{op}`.\n{purpose_of_op}",
                )[0]
            else:
                api_reference = self.api_reference_retriever.predict(query=op)
            ops_with_api_reference.append(
                OpWithAPIReference(
                    op=op,
                    api_reference=api_reference["text"],
                    api_reference_path=api_reference["file_path"],
                )
            )
        return ops_with_api_reference

    @weave.op()
    def predict(
        self, code_snippet: str, seed: Optional[int] = None, max_retries: int = 3
    ) -> Dict[str, List[OpWithAPIReference]]:
        ops = self.extract_operations(
            code_snippet=code_snippet, seed=seed, max_retries=max_retries
        )
        return {"retrieved_ops_with_references": self.retrieve_api_references(ops=ops)}
