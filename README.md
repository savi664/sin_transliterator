# sin_transliterator

[![PyPI version](https://img.shields.io/pypi/v/sin_transliterator)](https://pypi.org/project/sin_transliterator/)
[![CI](https://github.com/savi664/sin_transliterate/actions/workflows/ci.yml/badge.svg)](https://github.com/savi664/sin_transliterate/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/sin_transliterator)](https://pypi.org/project/sin_transliterator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Python package for **Sinhala phonetic and adhoc transliteration** — converting romanised Sinhala (Singlish) to Sinhala Unicode script. Backed by four fine-tuned models hosted on HuggingFace, with full support for Sinhala-English code-mixed input.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Batch Transliteration](#batch-transliteration)
- [Model Selection Guide](#model-selection-guide)
- [VRAM & Hardware Requirements](#vram--hardware-requirements)
- [API Reference](#api-reference)
- [Error Handling](#error-handling)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

`sin_transliterator` exposes a single class — `SinTransliterator` — that resolves the correct model at runtime based on two parameters: the model architecture family and whether the input contains code-mixed text.

Four fine-tuned models underpin the package:

| Model | Architecture | Code-Mix Support | HuggingFace |
|---|---|---|---|
| Small100 Base | Seq2Seq Transformer | No | [savinugunarathna/Small100-Singlish-Sinhala-Merged](https://huggingface.co/savinugunarathna/Small100-Singlish-Sinhala-Merged) |
| Small100 Code-Mix | Seq2Seq Transformer | Yes | [savinugunarathna/small-100-Singlish-Sinhala-CodeMix](https://huggingface.co/savinugunarathna/small-100-Singlish-Sinhala-CodeMix) |
| Gemma3 Base | Causal LLM | No | [savinugunarathna/Gemma3-Singlish-Sinhala-Merged](https://huggingface.co/savinugunarathna/Gemma3-Singlish-Sinhala-Merged) |
| Gemma3 Code-Mix | Causal LLM | Yes | [savinugunarathna/Gemma3-Singlish-Sinhala-CodeMix](https://huggingface.co/savinugunarathna/Gemma3-Singlish-Sinhala-CodeMix) |

---

## Installation

```bash
pip install sin_transliterator
```

---

## Quickstart

Downloads model weights on first run and caches them locally. Works on any machine — GPU is used automatically if available, otherwise falls back to CPU.

```python
from sin_transliterator import SinTransliterator

# Pure Sinhala input
t = SinTransliterator(model="transformer", contains_code_mix=False)
print(t.transliterate("mama yanawa"))

# Code-mixed input (Sinhala + English)
t_mix = SinTransliterator(model="transformer", contains_code_mix=True)
print(t_mix.transliterate("mama office ekkata yanawa"))

# Force a specific device
t = SinTransliterator(model="llm", contains_code_mix=True, device="cuda")
print(t.transliterate("mama office ekkata yanawa"))
```

Recommended on Kaggle, Google Colab, or any machine with a GPU for best performance. CPU inference is also fully supported.

---

## Batch Transliteration

For converting large amounts of text, use `transliterate_batch()` instead of calling `transliterate()` in a loop. It feeds all inputs through the model in a single forward pass, making it significantly faster — especially on a GPU.

```python
from sin_transliterator import SinTransliterator

t = SinTransliterator(model="transformer", contains_code_mix=False)

results = t.transliterate_batch([
    "mama yanawa",
    "kohomada",
    "mama iskole yanawa",
    "moko wenne"
])

for r in results:
    print(r)
```

For very large datasets, split into chunks to avoid running out of memory:

```python
import pandas as pd
from sin_transliterator import SinTransliterator

df = pd.read_csv("your_file.csv")
t = SinTransliterator(model="transformer", contains_code_mix=False)

chunk_size = 32
results = []
for i in range(0, len(df), chunk_size):
    chunk = df["singlish"][i:i + chunk_size].tolist()
    results.extend(t.transliterate_batch(chunk))

df["sinhala"] = results
df.to_csv("output.csv", index=False)
```

---

## Model Selection Guide

| Scenario | `model` | `contains_code_mix` |
|---|---|---|
| Pure Sinhala, fast inference | `"transformer"` | `False` |
| Pure Sinhala, higher quality | `"llm"` | `False` |
| Sinhala + English mixed, fast | `"transformer"` | `True` |
| Sinhala + English mixed, higher quality | `"llm"` | `True` |

**When in doubt, start with `model="transformer"`.** It's faster, lighter, and performs well on the vast majority of standard Singlish input. Reach for `model="llm"` when output quality on complex or ambiguous inputs matters more than speed.

---

## VRAM & Hardware Requirements

Both models are deliberately compact — well within reach of consumer hardware.

| Model | Parameters | VRAM (estimated) | CPU Inference |
|---|---|---|---|
| Small100 (transformer) | ~300M | ~0.75 GB | Viable |
| Gemma3 (llm) | ~300M | ~0.60 GB | Viable |

Any GPU with 2 GB+ VRAM handles both models comfortably. CPU inference is slower but fully functional — expect a few seconds per inference on a modern CPU.

---

## API Reference

### `SinTransliterator`

```python
SinTransliterator(
    model="transformer",        # "transformer" | "llm"
    contains_code_mix=False,    # bool
    device=None,                # str — e.g. "cuda", "cpu", "mps"
    cache_dir=None              # str — custom cache path
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"transformer"` | Architecture family. `"transformer"` uses the fine-tuned Small100 seq2seq model; `"llm"` uses the fine-tuned Gemma3 causal model. |
| `contains_code_mix` | `bool` | `False` | When `True`, loads the variant fine-tuned on Sinhala-English code-mixed data. |
| `device` | `str` | `None` | PyTorch device string. Auto-detected if not set (`"cuda"` if available, else `"cpu"`). |
| `cache_dir` | `str` | `None` | Directory for caching downloaded weights. Defaults to `~/.cache/huggingface/`. |

#### `.transliterate(text, max_new_tokens=256)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | — | Romanised Sinhala or code-mixed input text. |
| `max_new_tokens` | `int` | `256` | Maximum number of tokens to generate. |

Returns a `str` containing the Sinhala Unicode output.

#### `.transliterate_batch(texts, max_new_tokens=256)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `texts` | `list of str` | — | List of romanised Sinhala or code-mixed input strings. |
| `max_new_tokens` | `int` | `256` | Maximum number of tokens to generate per input. |

Returns a `list of str` containing Sinhala Unicode outputs in the same order as the input. Significantly faster than calling `.transliterate()` in a loop for large inputs.

---

## Error Handling

```python
from sin_transliterator import SinTransliterator
from sin_transliterator.exceptions import (
    InvalidModelError,
    ModelLoadError,
    TransliterationError,
)

try:
    t = SinTransliterator(model="transformer", contains_code_mix=False)
    result = t.transliterate("mama yanawa")
except InvalidModelError as e:
    print(f"Configuration error: {e}")
except ModelLoadError as e:
    print(f"Model load failed: {e}")
except TransliterationError as e:
    print(f"Inference error: {e}")
```

---

## Acknowledgements

This package would not exist without the following contributions:

**Deshan Sumanathilaka** — for building and curating the core Singlish-Sinhala transliteration datasets used to train the base models.
- HuggingFace: [deshanksuman](https://huggingface.co/deshanksuman)
- Academic: [deshan.s@iit.ac.lk](mailto:deshan.s@iit.ac.lk)
- Personal: [deshanitacademy@gmail.com](mailto:deshanitacademy@gmail.com)

**Rukshan Dias** — for contributing the Sinhala-English code-mix dataset that made the code-mix model variants possible.
- LinkedIn: [rukshan-dias](https://www.linkedin.com/in/rukshan-dias/)
- Academic: [rukshan.20210046@iit.ac.lk](mailto:rukshan.20210046@iit.ac.lk)

---

## Contact

**Savinu Linath Gunarathna**
- HuggingFace: [savinugunarathna](https://huggingface.co/savinugunarathna)
- Email: [savinugunarathna4@gmail.com](mailto:savinugunarathna4@gmail.com)
- GitHub: [savi664](https://github.com/savi664)

---

*Built with love for the Sinhala NLP community.*