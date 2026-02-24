from __future__ import annotations

from typing import List

from .exceptions import InvalidModelError, ModelLoadError, TransliterationError
from .model_registry import ModelType, resolve


class SinTransliterator:
    """
    Sinhala transliterator backed by fine-tuned HuggingFace models.

    Parameters
    ----------
    model : {"transformer", "llm"}
        Architecture family to use.
    contains_code_mix : bool
        Load the code-mix capable variant when True.
    device : str, optional
        PyTorch device string. Auto-detected if not set.
    cache_dir : str, optional
        Local cache directory for downloaded model weights.

    Examples
    --------
    >>> t = SinTransliterator(model="transformer", contains_code_mix=False)
    >>> t.transliterate("mama yanawa")
    >>> t.transliterate_batch(["mama yanawa", "kohomada"])
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
        self._spec = resolve(model, contains_code_mix)
        self._load_local(device, cache_dir)

    def _load_local(self, device: str | None, cache_dir: str | None) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "sin_transliterate requires 'transformers' and 'torch'. "
                "Install with: pip install sin_transliterate[local]"
            ) from e

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._spec.repo_id, cache_dir=cache_dir
            )
            if self._spec.architecture == "seq2seq":
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self._spec.repo_id, cache_dir=cache_dir
                ).to(self._device)  # type: ignore[arg-type]
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._spec.repo_id, cache_dir=cache_dir
                ).to(self._device)  # type: ignore[arg-type]
            self._model.eval()

        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load '{self._spec.repo_id}' from HuggingFace Hub.\n"
                f"Original error: {exc}"
            ) from exc

    def _run_inference(self, texts: List[str], max_new_tokens: int) -> List[str]:
        import torch
        try:
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self._device)

            with torch.no_grad():
                if self._spec.architecture == "seq2seq":
                    output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
                else:
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

            return [
                self._tokenizer.decode(ids[1:] if self._spec.architecture == "seq2seq" else ids,
                                       skip_special_tokens=True)
                for ids in output_ids
            ]

        except Exception as exc:
            raise TransliterationError(
                f"Inference failed.\nOriginal error: {exc}"
            ) from exc

    def transliterate(self, text: str, max_new_tokens: int = 256) -> str:
        """
        Transliterate a single romanised Sinhala string to Sinhala Unicode.

        Parameters
        ----------
        text : str
            Input romanised or code-mixed text.
        max_new_tokens : int
            Maximum tokens to generate. Defaults to 256.

        Returns
        -------
        str
            Sinhala Unicode output.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        return self._run_inference([text], max_new_tokens)[0]

    def transliterate_batch(self, texts: List[str], max_new_tokens: int = 256) -> List[str]:
        """
        Transliterate a list of romanised Sinhala strings to Sinhala Unicode
        in a single batched forward pass.

        Parameters
        ----------
        texts : list of str
            Input romanised or code-mixed strings.
        max_new_tokens : int
            Maximum tokens to generate per input. Defaults to 256.

        Returns
        -------
        list of str
            Sinhala Unicode outputs in the same order as the input list.
        """
        if not isinstance(texts, list) or not texts:
            raise ValueError("Input must be a non-empty list of strings.")
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("All items in the list must be non-empty strings.")

        return self._run_inference(texts, max_new_tokens)

    def __repr__(self) -> str:
        return (
            f"SinTransliterator("
            f"model={self._model_type!r}, "
            f"contains_code_mix={self._contains_code_mix})"
        )
