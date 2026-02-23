# sin_transliterate

[![PyPI version](https://badge.fury.io/py/sin_transliterate.svg)](https://badge.fury.io/py/sin_transliterate)
[![CI](https://github.com/savi664/sin_transliterate/actions/workflows/ci.yml/badge.svg)](https://github.com/savi664/sin_transliterate/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/sin_transliterate)](https://pypi.org/project/sin_transliterate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Python package for **Sinhala phonetic transliteration** — converting romanised Sinhala (Singlish) to Sinhala Unicode script. Backed by four fine-tuned models hosted on HuggingFace, with full support for Sinhala-English code-mixed input and two inference modes: local and API.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Model Selection Guide](#model-selection-guide)
- [Inference Modes](#inference-modes)
- [VRAM & Hardware Requirements](#vram--hardware-requirements)
- [API Reference](#api-reference)
- [Error Handling](#error-handling)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

`sin_transliterate` exposes a single class — `SinTransliterator` — that resolves the correct model at runtime based on two parameters: the model architecture family and whether the input contains code-mixed text.

Four fine-tuned models underpin the package:

| Model | Architecture | Code-Mix Support | HuggingFace |
|---|---|---|---|
| Small100 Base | Seq2Seq Transformer | No | [savinugunarathna/Small100-Singlish-Sinhala-Merged](https://huggingface.co/savinugunarathna/Small100-Singlish-Sinhala-Merged) |
| Small100 Code-Mix | Seq2Seq Transformer | Yes | [savinugunarathna/small-100-Singlish-Sinhala-CodeMix](https://huggingface.co/savinugunarathna/small-100-Singlish-Sinhala-CodeMix) |
| Gemma3 Base | Causal LLM | No | [savinugunarathna/Gemma3-Singlish-Sinhala-Merged](https://huggingface.co/savinugunarathna/Gemma3-Singlish-Sinhala-Merged) |
| Gemma3 Code-Mix | Causal LLM | Yes | [savinugunarathna/Gemma3-Singlish-Sinhala-CodeMix](https://huggingface.co/savinugunarathna/Gemma3-Singlish-Sinhala-CodeMix) |

---

## Installation

**API mode only** — no GPU or model download required:
```bash
pip install sin_transliterate
```

**Local inference** — downloads and runs models on your own machine:
```bash
pip install sin_transliterate[local]
```

---

## Quickstart

### API Mode (recommended for most users)

No model download. No GPU. Just a free HuggingFace token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```python
from sin_transliterate import SinTransliterator

# Pure Sinhala input
t = SinTransliterator(
    model="transformer",
    contains_code_mix=False,
    mode="api",
    hf_token="hf_your_token_here"
)
print(t.transliterate("mama yanawa"))

# Code-mixed input (Sinhala + English)
t_mix = SinTransliterator(
    model="transformer",
    contains_code_mix=True,
    mode="api",
    hf_token="hf_your_token_here"
)
print(t_mix.transliterate("mama office ekkata yanawa"))
```

### Local Mode

Downloads model weights to your machine and runs inference locally.

```python
from sin_transliterate import SinTransliterator

# Uses GPU automatically if available, falls back to CPU
t = SinTransliterator(
    model="transformer",
    contains_code_mix=False,
    mode="local"
)
print(t.transliterate("mama yanawa"))

# Force a specific device
t = SinTransliterator(
    model="llm",
    contains_code_mix=True,
    mode="local",
    device="cuda"
)
print(t.transliterate("mama office ekkata yanawa"))
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

## Inference Modes

### `mode="api"` (default recommendation)

- Calls the HuggingFace Inference API — no weights downloaded locally
- Requires a free HuggingFace account and API token
- Works on any machine, including low-end hardware and Colab free tier
- Subject to HuggingFace rate limits on the free tier

### `mode="local"`

- Downloads model weights to `~/.cache/huggingface/` on first run
- Subsequent runs load from cache — no repeated downloads
- No rate limits, fully offline after initial download
- Requires `pip install sin_transliterate[local]`

---

## VRAM & Hardware Requirements

Both models are deliberately compact — well within reach of consumer hardware.

| Model | Parameters | VRAM (estimated) | CPU Inference |
|---|---|---|---|
| Small100 (transformer) | ~600M | ~0.75 GB | Viable |
| Gemma3 (llm) | ~500M | ~0.60 GB | Viable |

Any GPU with 2 GB+ VRAM handles both models comfortably. CPU inference is slower but fully functional — expect a few seconds per inference on a modern CPU.

---

## API Reference

### `SinTransliterator`

```python
SinTransliterator(
    model="transformer",        # "transformer" | "llm"
    contains_code_mix=False,    # bool
    mode="local",               # "local" | "api"
    hf_token=None,              # str — required when mode="api"
    device=None,                # str — e.g. "cuda", "cpu", "mps" (local mode only)
    cache_dir=None              # str — custom cache path (local mode only)
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"transformer"` | Architecture family. `"transformer"` uses the fine-tuned Small100 seq2seq model; `"llm"` uses the fine-tuned Gemma3 causal model. |
| `contains_code_mix` | `bool` | `False` | When `True`, loads the variant fine-tuned on Sinhala-English code-mixed data. |
| `mode` | `str` | `"local"` | `"local"` runs inference on your machine. `"api"` calls the HuggingFace Inference API. |
| `hf_token` | `str` | `None` | HuggingFace API token. Mandatory when `mode="api"`. |
| `device` | `str` | `None` | PyTorch device string. Auto-detected if not set (`"cuda"` if available, else `"cpu"`). Only applies to `mode="local"`. |
| `cache_dir` | `str` | `None` | Directory for caching downloaded weights. Defaults to `~/.cache/huggingface/`. Only applies to `mode="local"`. |

#### `.transliterate(text, max_new_tokens=256)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | — | Romanised Sinhala or code-mixed input text. |
| `max_new_tokens` | `int` | `256` | Maximum number of tokens to generate. |

Returns a `str` containing the Sinhala Unicode output.

---

## Error Handling

```python
from sin_transliterate import SinTransliterator
from sin_transliterate.exceptions import (
    InvalidModelError,
    ModelLoadError,
    TransliterationError,
)

try:
    t = SinTransliterator(model="transformer", contains_code_mix=False, mode="api", hf_token="hf_...")
    result = t.transliterate("mama yanawa")
except InvalidModelError as e:
    # Invalid model or mode parameter passed
    print(f"Configuration error: {e}")
except ModelLoadError as e:
    # Model failed to download or initialise (local mode)
    print(f"Model load failed: {e}")
except TransliterationError as e:
    # Inference failed on the given input
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