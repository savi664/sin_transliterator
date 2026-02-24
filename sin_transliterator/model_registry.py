from dataclasses import dataclass
from typing import Literal

ModelType = Literal["transformer", "llm"]


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    architecture: str
    supports_code_mix: bool


_REGISTRY: dict[tuple[ModelType, bool], ModelSpec] = {
    ("transformer", False): ModelSpec(
        repo_id="savinugunarathna/Small100-Singlish-Sinhala-Merged",
        architecture="seq2seq",
        supports_code_mix=False,
    ),
    ("transformer", True): ModelSpec(
        repo_id="savinugunarathna/small-100-Singlish-Sinhala-CodeMix",
        architecture="seq2seq",
        supports_code_mix=True,
    ),
    ("llm", False): ModelSpec(
        repo_id="savinugunarathna/Gemma3-Singlish-Sinhala-Merged",
        architecture="causal",
        supports_code_mix=False,
    ),
    ("llm", True): ModelSpec(
        repo_id="savinugunarathna/Gemma3-Singlish-Sinhala-CodeMix",
        architecture="causal",
        supports_code_mix=True,
    ),
}


def resolve(model: ModelType, contains_code_mix: bool) -> ModelSpec:
    key = (model, contains_code_mix)
    if key not in _REGISTRY:
        raise KeyError(
            f"No model registered for model={model!r}, contains_code_mix={contains_code_mix}"
        )
    return _REGISTRY[key]
