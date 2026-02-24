import pytest
import torch
from unittest.mock import patch, MagicMock
from sin_transliterator import SinTransliterator
from sin_transliterator.exceptions import InvalidModelError


@pytest.fixture
def mock_transformer_env():
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.decode.return_value = "මම යනවා"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

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
        t = SinTransliterator(model="transformer", contains_code_mix=False)
        with patch.object(t, "_run_inference", return_value=["මම යනවා"]):
            result = t.transliterate("mama yanawa")
        assert isinstance(result, str)
        assert result == "මම යනවා"

    def test_batch_returns_list(self, mock_transformer_env):
        t = SinTransliterator(model="transformer", contains_code_mix=False)
        with patch.object(t, "_run_inference", return_value=["මම යනවා", "කොහොමද"]):
            results = t.transliterate_batch(["mama yanawa", "kohomada"])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_batch_empty_list_raises(self, mock_transformer_env):
        t = SinTransliterator(model="transformer", contains_code_mix=False)
        with pytest.raises(ValueError):
            t.transliterate_batch([])

    def test_batch_invalid_item_raises(self, mock_transformer_env):
        t = SinTransliterator(model="transformer", contains_code_mix=False)
        with pytest.raises(ValueError):
            t.transliterate_batch(["mama yanawa", ""])