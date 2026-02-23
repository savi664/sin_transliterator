# sin_transliterate/core.py

from __future__ import annotations

from typing import Literal

from .exceptions import InvalidModelError, ModelLoadError, TransliterationError
from .model_registry import ModelType, resolve

try:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )
    import torch
except ImportError as e:
    raise ImportError(
        "sin_transliterate requires 'transformers' and 'torch'. "
        "Install them with: pip install sin_transliterate"
    ) from e


class SinTransliterator:
    """
    Sinhala transliterator backed by fine-tuned HuggingFace models.

    Parameters
    ----------
    model : {"transformer", "llm"}
        Architecture family to load. "transformer" selects the fine-tuned
        seq2seq model; "llm" selects the causal language model variant.
    contains_code_mix : bool
        When True, loads the variant fine-tuned to handle Sinhala/English
        code-mixed input. Defaults to False.
    device : str, optional
        PyTorch device string (e.g. "cpu", "cuda", "mps"). Defaults to
        automatic detection.
    cache_dir : str, optional
        Local directory for caching downloaded model weights. Defaults to
        the HuggingFace Hub default (~/.cache/huggingface).

    Examples
    --------
    >>> t = SinTransliterator(model="transformer", contains_code_mix=False)
    >>> t.transliterate("mama yanawa")
    'මම යනවා'

    >>> t_mix = SinTransliterator(model="llm", contains_code_mix=True)
    >>> t_mix.transliterate("mama office yanawa")
    'මම ඔෆිස් යනවා'
    """

    VALID_MODELS = ("transformer", "llm")

    def __init__(
            self,
            model: ModelType = "transformer",
            contains_code_mix: bool = False,
            device: str | None = None,
            cache_dir: str | None = None,
    ) -> None:
        if model not in self.VALID_MODELS:
            raise InvalidModelError(
                f"model={model!r} is not valid. Choose from {self.VALID_MODELS}."
            )

        self._model_type = model
        self._contains_code_mix = contains_code_mix
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        spec = resolve(model, contains_code_mix)
        self._spec = spec

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                spec.repo_id, cache_dir=cache_dir
            )

            if spec.architecture == "seq2seq":
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    spec.repo_id, cache_dir=cache_dir
                ).to(self._device)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    spec.repo_id, cache_dir=cache_dir
                ).to(self._device)

            self._model.eval()

        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model '{spec.repo_id}' from HuggingFace Hub. "
                f"Check your internet connection and that the repo is accessible.\n"
                f"Original error: {exc}"
            ) from exc

    def transliterate(self, text: str, max_new_tokens: int = 256) -> str:
        """
        Transliterate romanised Sinhala text to Sinhala Unicode script.

        Parameters
        ----------
        text : str
            Input romanised or code-mixed text.
        max_new_tokens : int
            Maximum number of tokens to generate. Defaults to 256.

        Returns
        -------
        str
            Sinhala Unicode output.

        Raises
        ------
        TransliterationError
            If inference fails for any reason.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        try:
            inputs = self._tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)

            with torch.no_grad():
                if self._spec.architecture == "seq2seq":
                    output_ids = self._model.generate(
                        **inputs, max_new_tokens=max_new_tokens
                    )
                else:
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

            return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

        except Exception as exc:
            raise TransliterationError(
                f"Inference failed on input: {text!r}\nOriginal error: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"SinTransliterator("
            f"model={self._model_type!r}, "
            f"contains_code_mix={self._contains_code_mix}, "
            f"device={self._device!r})"
        )
