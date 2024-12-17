import litellm
import weave

from .op_extraction import DocumentationRetreiver, OpExtractor
from .prompts import FRAMEWORK_IDENTIFICATION_PROMPT
from .retrieval import HeuristicRetreiver
from .utils import SupportedFrameworks, get_structured_output_from_completion


class FrameworkIdentificationModel(weave.Model):
    model_name: str

    @weave.op()
    def predict(self, code_snippet: str) -> str:
        completion = litellm.completion(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": FRAMEWORK_IDENTIFICATION_PROMPT,
                },
                {
                    "role": "user",
                    "content": code_snippet,
                },
            ],
            response_format=SupportedFrameworks,
        )
        return get_structured_output_from_completion(
            completion,
            response_format=SupportedFrameworks,
        ).frameworks


class TranslationAgent(weave.Model):
    model_name: str
    verbose: bool = True

    @weave.op()
    def predict(
        self,
        code_snippet: str,
        # target_framework: str,
    ):
        # identify source framework
        framework_identification_model = FrameworkIdentificationModel(
            model_name=self.model_name
        )
        source_framework = framework_identification_model.predict(
            code_snippet=code_snippet
        )

        # initialise retriever
        source_retriever: DocumentationRetreiver = HeuristicRetreiver(
            framework=source_framework,
        )

        # Op Extraction
        source_op_extractor = OpExtractor(
            model_name=self.model_name,
            api_reference_retriever=source_retriever,
            verbose=self.verbose,
        )
        source_ops = source_op_extractor.predict(code_snippet=code_snippet)  # noqa: F841
        return source_ops
