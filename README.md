# Griffin

PyTorch implementation of the hybrid Griffin language-model architecture from
[the Griffin paper](https://arxiv.org/pdf/2402.19427.pdf). The default temporal
schedule repeats two RG-LRU recurrent blocks followed by one causal local-MQA
block. Input embeddings and the vocabulary head share weights.

On CUDA systems with the optional `triton` package installed, `scan_mode="auto"`
uses a fused linear-work RG-LRU scan for both forward and backward. CPU systems
use the sequential reference path, while `scan_mode="associative"` remains
available as a portable parallel implementation. Use `scan_mode="fused"` to
require the Triton backend and receive an explicit error when it is unavailable.

Run the behavioral suite:

```powershell
python -m pytest -q
```

Run an offline end-to-end training smoke test:

```powershell
python train.py --smoke-test --max-steps 2
```

The regular trainer defaults to the paper's 100M-scale dimensions, a 2048-token
context, and a 1024-token attention window. MassiveText is not public, so the
dataset and tokenizer are configurable Hugging Face inputs rather than an exact
copy of the paper's corpus:

```powershell
python train.py --dataset-name wikitext --dataset-config wikitext-103-raw-v1 --tokenizer-name gpt2
```
