
# Gemma 3

A minimal [Gemma 3 270M IT](https://huggingface.co/google/gemma-3-270m-it) decoder-only implementation in JAX.

## Get started
1. Clone the repository
```bash
git clone https://github.com/lutzroeder/models
```
2. Install dependencies
```bash
pip install jax jaxlib numpy sentencepiece huggingface_hub
```
3. Authenticate as Gemma 3 is a gated model
```bash
huggingface-cli login
```
4. Run the model
```bash
python models/gemma3/gemma3.py "What is the capital of France?"
```
```
The capital of France is Paris.
```
