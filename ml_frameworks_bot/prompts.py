from typing import get_args

from .utils import SupportedFrameworks


OP_EXTRACTION_TMPL: str = """
You are an experienced machine learning engineer expert in python and `{framework}`.
You are suppossed to think step-by-step about all the unique `{framework}` operations,
layers, and functions from a given snippet of code.

Here are some rules:
1. All functions and classes that are imported from `{framework}` should be considered
    to be `{framework}` operations.
2. `import` statements don't count as separate statements.
3. If there are nested `{framework}` operations, you should extract all the operations
    that are present inside the parent operation.
4. You should simply return the names of the ops and not the entire statement itself.
5. You must also consider all member functions of a `{framework}` operation as separate
    `{framework}` operations. But arguments should not be considered as separate
    `{framework}` operations. Also calling the operation should not be considered
    as a separate operation.
6. Ensure that the names of the ops consist of the entire `{framework}` namespace,
    starting with `{framework}`.
"""

FRAMEWORK_IDENTIFICATION_PROMPT: str = f"""
You are an experienced machine learning engineer expert in python, and the following
frameworks: {", ".join(get_args(SupportedFrameworks.__annotations__["frameworks"]))}.
You will be provided with a code snippet and you need to identify from the
aforementioned frameworks the code snippet belongs to.

You must follow the following rules:
1. If keras is imported using `from tensorflow import keras`, then the framework
    is keras2. Also, if tensorflow is imported using `import tensorflow as tf`
    and then keras is used as `tf.keras`, then the framework is keras2.
2. If keras is imported using `import keras` without it being imported
    from tensorflow, then the framework is keras3.
"""
