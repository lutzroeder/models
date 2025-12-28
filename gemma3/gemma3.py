import functools
import json
import math
import os
import struct
import sys
import urllib.request

import jax
import jax.numpy as jnp
import numpy
import sentencepiece


def load_safetensors(path):
    params = {}
    with open(path, 'rb') as file:
        header_size = struct.unpack('<Q', file.read(8))[0]
        content = file.read(header_size)
        header = json.loads(content)
        data = file.read()
    dtypes = {'F32': numpy.float32, 'F16': numpy.float16, 'BF16': '>u2'}
    for name, info in header.items():
        if name != '__metadata__':
            start, end = info['data_offsets']
            dtype = dtypes[info['dtype']]
            arr = numpy.frombuffer(data[start:end], dtype=dtype).reshape(info['shape'])
            if info['dtype'] == 'BF16':
                params[name] = jnp.array(arr.view(numpy.uint16)).view(jnp.bfloat16)
            else:
                params[name] = jnp.array(arr)
    return params

def download(model_id, file):
    path = f'{os.path.dirname(__file__)}/{file}'
    if not os.path.isfile(path):
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if not token:
            token_path = os.path.expanduser('~/.cache/huggingface/token')
            if os.path.exists(token_path):
                with open(token_path) as f:
                    token = f.read().strip()
        url = f'https://huggingface.co/{model_id}/resolve/main/{file}'
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request) as response:
            total_size = int(response.headers['content-length'])
            with open(path, 'wb') as f:
                downloaded = 0
                while block := response.read(8192):
                    f.write(block)
                    downloaded += len(block)
                    print(f"\r\033[KDownloading '{file}' ({100*downloaded//total_size}%)", end='', flush=True)
        print()
    return path

class Tokenizer:

    def __init__(self, path):
        self.sp = sentencepiece.SentencePieceProcessor(path)
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.end_turn_id = self.sp.PieceToId('<end_of_turn>')

    def encode(self, text, add_bos=False):
        return self.sp.Encode(text, add_bos=add_bos)

    def decode(self, ids):
        return self.sp.Decode(list(ids))

class Gemma3:

    def __init__(self, config, params):
        self.rng_key = jax.random.PRNGKey(42)
        self.params = params
        self.num_layers = config['num_hidden_layers']
        self.hidden = config['hidden_size']
        self.heads = config['num_attention_heads']
        self.kv_heads = config['num_key_value_heads']
        self.head_dim = config['head_dim']
        self.rms_norm_eps = config['rms_norm_eps']
        self.layer_types = tuple(config['layer_types'])
        self.sliding_window = config['sliding_window']
        self.attn_scale = 1.0 / math.sqrt(config['query_pre_attn_scalar'])
        self.max_cache_len = config['max_position_embeddings']
        self.embed = params['model.embed_tokens.weight']
        def rope_freqs(dim, max_pos, theta=10000.0):
            freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
            t = jnp.arange(max_pos, dtype=jnp.float32)
            freqs = jnp.outer(t, freqs)
            return jnp.cos(jnp.concatenate([freqs, freqs], axis=-1)), jnp.sin(jnp.concatenate([freqs, freqs], axis=-1))
        self.local_rope = rope_freqs(self.head_dim, self.max_cache_len, config['rope_local_base_freq'])
        self.global_rope = rope_freqs(self.head_dim, self.max_cache_len, config['rope_theta'])

    @staticmethod
    @functools.partial(jax.jit, static_argnames=['num_layers', 'heads', 'kv_heads', 'head_dim', 'hidden', 'rms_norm_eps', 'attn_scale', 'layer_types', 'sliding_window', 'max_cache_len'])
    def forward_jit(input_ids, params, embed, local_rope, global_rope, kv_cache, pos, num_layers, heads, kv_heads, head_dim, hidden, rms_norm_eps, attn_scale, layer_types, sliding_window, max_cache_len):
        def rms_norm(x, weight, eps=1e-6):
            x32 = x.astype(jnp.float32)
            x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 ** 2, axis=-1, keepdims=True) + eps)
            return (x32 * (1.0 + weight.astype(jnp.float32))).astype(x.dtype)
        def apply_rope(x, cos, sin, position_ids):
            cos = cos[position_ids][:, :, None, :]
            sin = sin[position_ids][:, :, None, :]
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return x * cos + jnp.concatenate([-x2, x1], axis=-1) * sin
        def attention(q, k, v, mask, scale):
            scores = jnp.einsum('bthd,bshd->bhts', q, k) * scale
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
            return jnp.einsum('bhts,bshd->bthd', jax.nn.softmax(scores, axis=-1), v)
        def repeat_kv(x, n):
            return x if n == 1 else jnp.repeat(x, n, axis=2)
        bs, seq_len = input_ids.shape
        position_ids = (pos + jnp.arange(seq_len, dtype=jnp.int32))[None, :]
        x = embed[input_ids] * math.sqrt(hidden)
        new_cache = []
        for i in range(num_layers):
            p = f'model.layers.{i}'
            is_sliding = layer_types[i] == 'sliding_attention'
            residual = x
            x = rms_norm(x, params[f'{p}.input_layernorm.weight'], rms_norm_eps)
            q = (x @ params[f'{p}.self_attn.q_proj.weight'].T).reshape(bs, seq_len, heads, head_dim)
            k = (x @ params[f'{p}.self_attn.k_proj.weight'].T).reshape(bs, seq_len, kv_heads, head_dim)
            v = (x @ params[f'{p}.self_attn.v_proj.weight'].T).reshape(bs, seq_len, kv_heads, head_dim)
            q = rms_norm(q, params[f'{p}.self_attn.q_norm.weight'], rms_norm_eps)
            k = rms_norm(k, params[f'{p}.self_attn.k_norm.weight'], rms_norm_eps)
            rope = local_rope if is_sliding else global_rope
            q = apply_rope(q, rope[0], rope[1], position_ids)
            k = apply_rope(k, rope[0], rope[1], position_ids)
            k_cache = jax.lax.dynamic_update_slice(kv_cache[i][0], k.astype(jnp.bfloat16), (0, pos, 0, 0))
            v_cache = jax.lax.dynamic_update_slice(kv_cache[i][1], v.astype(jnp.bfloat16), (0, pos, 0, 0))
            new_cache.append((k_cache, v_cache))
            kv_len = pos + seq_len
            if is_sliding:
                start = jnp.maximum(0, kv_len - sliding_window)
                k_use = jax.lax.dynamic_slice(k_cache, (0, start, 0, 0), (bs, sliding_window, kv_heads, head_dim))
                v_use = jax.lax.dynamic_slice(v_cache, (0, start, 0, 0), (bs, sliding_window, kv_heads, head_dim))
                q_pos = pos + jnp.arange(seq_len)
                k_pos = start + jnp.arange(sliding_window)
                mask = (k_pos[None, :] <= q_pos[:, None]) & (k_pos[None, :] < kv_len) & (k_pos[None, :] >= q_pos[:, None] - sliding_window + 1)
            else:
                k_use, v_use = k_cache, v_cache
                q_pos = pos + jnp.arange(seq_len)
                k_pos = jnp.arange(max_cache_len)
                mask = (k_pos[None, :] <= q_pos[:, None]) & (k_pos[None, :] < kv_len)
            mask = mask[None, None, :, :]
            k_rep = repeat_kv(k_use, heads // kv_heads)
            v_rep = repeat_kv(v_use, heads // kv_heads)
            attn_out = attention(q, k_rep, v_rep, mask, attn_scale).reshape(bs, seq_len, heads * head_dim)
            attn_out = attn_out @ params[f'{p}.self_attn.o_proj.weight'].T
            x = residual + rms_norm(attn_out, params[f'{p}.post_attention_layernorm.weight'], rms_norm_eps)
            residual = x
            x = rms_norm(x, params[f'{p}.pre_feedforward_layernorm.weight'], rms_norm_eps)
            gate = x @ params[f'{p}.mlp.gate_proj.weight'].T
            up = x @ params[f'{p}.mlp.up_proj.weight'].T
            x = jax.nn.gelu(gate, approximate=True) * up
            x = x @ params[f'{p}.mlp.down_proj.weight'].T
            x = residual + rms_norm(x, params[f'{p}.post_feedforward_layernorm.weight'], rms_norm_eps)
        x = rms_norm(x, params['model.norm.weight'], rms_norm_eps)
        return (x @ embed.T)[:, -1, :], new_cache

    def forward(self, input_ids, kv_cache, pos, temperature=0.7, top_k=40):
        if kv_cache is None:
            batch_size = input_ids.shape[0]
            shape = (batch_size, self.max_cache_len, self.kv_heads, self.head_dim)
            kv_cache = [(jnp.zeros(shape, jnp.bfloat16), jnp.zeros(shape, jnp.bfloat16)) for _ in range(self.num_layers)]
        logits, kv_cache = Gemma3.forward_jit(
            input_ids, self.params, self.embed, self.local_rope, self.global_rope,
            kv_cache, pos, self.num_layers, self.heads, self.kv_heads, self.head_dim,
            self.hidden, self.rms_norm_eps, self.attn_scale, self.layer_types,
            self.sliding_window, self.max_cache_len)
        logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            vals, _ = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
            logits = jnp.where(logits >= vals[:, -1:], logits, jnp.finfo(logits.dtype).min)
        self.rng_key, key = jax.random.split(self.rng_key)
        return (int(jnp.argmax(logits[0])) if temperature == 0 else int(jax.random.categorical(key, logits)[0])), kv_cache

prompt = 'What is the capital of France?' if len(sys.argv) <= 1 else sys.argv[1]
model_type = 'google/gemma-3-270m-it'
tokenizer = download(model_type, 'tokenizer.model')
config = download(model_type, 'config.json')
params = download(model_type, 'model.safetensors')
tokenizer = Tokenizer(tokenizer)
with open(config) as file:
    config = json.load(file)
params = load_safetensors(params)
model = Gemma3(config, params)
input_ids = jnp.array([tokenizer.encode(f'<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n', add_bos=True)], dtype=jnp.int32)
kv_cache = None
pos = 0
for _ in range(model.max_cache_len):
    token, kv_cache = model.forward(input_ids, kv_cache, pos, temperature=0.0)
    pos += input_ids.shape[1]
    if token in (tokenizer.eos_id, tokenizer.end_turn_id):
        break
    print(tokenizer.decode([token]), end='', flush=True)
    input_ids = jnp.array([[token]], dtype=jnp.int32)
print()
