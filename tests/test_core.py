import pytest
from unittest.mock import patch, MagicMock
from sin_transliterate import SinTransliterator
from sin_transliterate.exceptions import InvalidModelError, ModelLoadError


@pytest.fixture
def mock_transformer_env(monkeypatch):
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()
    mock_tokenizer.decode.return_value = "මම යනවා"

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=mock_model), \
         patch("torch.cuda.is_available", return_value=False):
        yield mock_tokenizer, mock_model


class TestInstantiation:
    def test_valid_transformer_no_codemix(self, mock_transformer_env):
        t = SinTransliterator(model="transformer", contains_code_mix=False)
        assert t is not None

    def test_invalid_model_raises(self):
        with pytest.raises(InvalidModelError, match="not valid"):
            SinTransliterator(model="bert")  # type: ignore

    def test_repr_contains_model_type(self, mock_transformer_env):
        t = SinTransliterator(model="transformer", contains_code_mix=False)
        assert "transformer" in repr(t)
        assert "False" in repr(t)


class TestTransliterate:
    def test_empty_string_raises_value_error(self, mock_transformer_env):
        t = SinTransliterator(model="transformer", contains_code_mix=False)
        with pytest.raises(ValueError):
            t.transliterate("")

    def test_whitespace_only_raises_value_error(self, mock_transformer_env):
        t = SinTransliterator(model="transformer", contains_code_mix=False)
        with pytest.raises(ValueError):
            t.transliterate("   ")

    def test_returns_string(self, mock_transformer_env):
        mock_tok, mock_mod = mock_transformer_env
        import torch

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tok.return_value = mock_inputs

        mock_mod.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tok.decode.return_value = "මම යනවා"

        t = SinTransliterator(model="transformer", contains_code_mix=False)
        result = t.transliterate("mama yanawa")
        assert isinstance(result, str)