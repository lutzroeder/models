
# Gemma 4

A minimal [Gemma 4 E2B IT](https://huggingface.co/google/gemma-4-E2B-it) decoder-only implementation in JAX.

## Get started
1. Clone the repository
```bash
git clone https://github.com/lutzroeder/models
```
2. Install dependencies
```bash
pip install jax jaxlib numpy tokenizers
```
3. Run the model
```bash
python models/gemma4/gemma4.py "What is the capital of France?"
```
```
The capital of France is Paris.
```
