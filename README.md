# Griffin

PyTorch implementation of the hybrid Griffin language-model architecture from
[the Griffin paper](https://arxiv.org/pdf/2402.19427.pdf). The default temporal
schedule repeats two RG-LRU recurrent blocks followed by one causal local-MQA
block. Input embeddings and the vocabulary head share weights.

Install the model package and development dependencies:

```powershell
python -m pip install -e ".[train,test]"
```

On supported CUDA environments, install the optional Triton backend with
`python -m pip install -e ".[triton]"`.

On CUDA systems with the optional `triton` package installed, `scan_mode="auto"`
runs a one-time forward/backward parity check before enabling the fused
linear-work RG-LRU scan. A compile or numerical failure emits a warning and
falls back to the portable associative scan. CPU systems use the sequential
reference path. Use `scan_mode="fused"` to require the Triton backend and expose
backend failures directly.

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
