# sin_transliterate

Sinhala phonetic transliteration from romanised script to Sinhala Unicode,
backed by fine-tuned seq2seq and LLM models with optional code-mix support.

## Installation
pip install sin_transliterate

## Quickstart
from sin_transliterate import SinTransliterator

# Basic phonetic transliteration
t = SinTransliterator(model="transformer", contains_code_mix=False)
print(t.transliterate("mama yanawa"))  # → මම යනවා

# Code-mix input (Sinhala + English)
t_mix = SinTransliterator(model="llm", contains_code_mix=True)
print(t_mix.transliterate("mama office ekkata yanawa"))

## Model Selection Guide

| Scenario | model | contains_code_mix |
|---|---|---|
| Pure Sinhala, fast inference | "transformer" | False |
| Pure Sinhala, higher quality | "llm" | False |
| Code-mixed, fast | "transformer" | True |
| Code-mixed, higher quality | "llm" | True |

## Error Handling
from sin_transliterate.exceptions import ModelLoadError, TransliterationError

try:
    result = t.transliterate(user_input)
except TransliterationError as e:
    print(f"Inference failed: {e}")