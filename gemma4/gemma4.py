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
import tokenizers


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
        self.tokenizer = tokenizers.Tokenizer.from_file(path)
        self.bos_id = 2
        self.eos_ids = {1, 106}

    def encode(self, text, add_bos=False):
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.bos_id] + ids
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(list(ids))


class Gemma4:

    def __init__(self, config, params):
        self.rng_key = jax.random.PRNGKey(42)
        self.params = params
        tc = config['text_config']
        self.num_layers = tc['num_hidden_layers']
        self.hidden = tc['hidden_size']
        self.heads = tc['num_attention_heads']
        self.kv_heads = tc['num_key_value_heads']
        self.head_dim = tc['head_dim']
        self.global_head_dim = tc['global_head_dim']
        self.rms_norm_eps = tc['rms_norm_eps']
        self.layer_types = tuple(tc['layer_types'])
        self.sliding_window = tc['sliding_window']
        self.max_cache_len = min(tc['max_position_embeddings'], 4096)
        self.final_logit_softcapping = tc.get('final_logit_softcapping')
        self.ple_dim = tc['hidden_size_per_layer_input']
        self.first_kv_shared_layer = self.num_layers - tc['num_kv_shared_layers']
        self.kv_donor_sliding = max(i for i in range(self.first_kv_shared_layer) if self.layer_types[i] == 'sliding_attention')
        self.kv_donor_full = max(i for i in range(self.first_kv_shared_layer) if self.layer_types[i] == 'full_attention')
        lm = 'model.language_model'
        self.embed = params[f'{lm}.embed_tokens.weight']
        self.embed_per_layer = params[f'{lm}.embed_tokens_per_layer.weight']
        self.layers = []
        for i in range(self.num_layers):
            prefix = f'{lm}.layers.{i}.'
            self.layers.append({k[len(prefix):].removesuffix('.weight'): v for k, v in params.items() if k.startswith(prefix)})
        rope_params = tc['rope_parameters']
        partial_rotary_factor = rope_params['full_attention'].get('partial_rotary_factor', 1.0)
        def rope_freqs(head_dim, max_pos, theta, rope_proportion=1.0):
            half_dim = head_dim // 2
            rope_angles = int(rope_proportion * half_dim)
            freq_exponents = (2.0 / head_dim) * jnp.arange(rope_angles, dtype=jnp.float32)
            timescale = theta ** freq_exponents
            if rope_angles < half_dim:
                timescale = jnp.concatenate([timescale, jnp.full(half_dim - rope_angles, jnp.inf)])
            t = jnp.arange(max_pos, dtype=jnp.float32)
            angles = t[:, None] / timescale[None, :]
            return jnp.cos(angles), jnp.sin(angles)
        self.local_rope = rope_freqs(self.head_dim, self.max_cache_len, rope_params['sliding_attention']['rope_theta'])
        self.global_rope = rope_freqs(self.global_head_dim, self.max_cache_len, rope_params['full_attention']['rope_theta'], rope_proportion=partial_rotary_factor)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=[
        'num_layers', 'heads', 'kv_heads', 'head_dim', 'global_head_dim', 'hidden',
        'rms_norm_eps', 'layer_types', 'sliding_window', 'max_cache_len',
        'first_kv_shared_layer', 'kv_donor_sliding', 'kv_donor_full',
        'ple_dim', 'final_logit_softcapping'
    ])
    def forward_jit(input_ids, params, layers, embed, embed_per_layer, local_rope, global_rope,
                    kv_cache, pos,
                    num_layers, heads, kv_heads, head_dim, global_head_dim, hidden,
                    rms_norm_eps, layer_types, sliding_window, max_cache_len,
                    first_kv_shared_layer, kv_donor_sliding, kv_donor_full,
                    ple_dim, final_logit_softcapping):
        def rms_norm(x, weight=None, eps=1e-6):
            x32 = x.astype(jnp.float32)
            x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 ** 2, axis=-1, keepdims=True) + eps)
            if weight is not None:
                x32 = x32 * weight.astype(jnp.float32)
            return x32.astype(x.dtype)
        def apply_rope(x, cos, sin, position_ids):
            cos = cos[position_ids][:, :, None, :]
            sin = sin[position_ids][:, :, None, :]
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
        def attention(q, k, v, mask):
            scores = jnp.einsum('bthd,bshd->bhts', q, k)
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
            return jnp.einsum('bhts,bshd->bthd', jax.nn.softmax(scores, axis=-1), v)
        def repeat_kv(x, n):
            return x if n == 1 else jnp.repeat(x, n, axis=2)
        bs, seq_len = input_ids.shape
        position_ids = (pos + jnp.arange(seq_len, dtype=jnp.int32))[None, :]
        lm = 'model.language_model'
        x = embed[input_ids] * math.sqrt(hidden)
        token_identity = embed_per_layer[input_ids] * math.sqrt(ple_dim)
        token_identity = token_identity.reshape(bs, seq_len, num_layers, ple_dim)
        context = (x @ params[f'{lm}.per_layer_model_projection.weight'].T) * (hidden ** -0.5)
        context = context.reshape(bs, seq_len, num_layers, ple_dim)
        context = rms_norm(context, params[f'{lm}.per_layer_projection_norm.weight'], rms_norm_eps)
        per_layer_inputs = (context + token_identity) * (2.0 ** -0.5)
        new_cache = []
        for i in range(num_layers):
            lp = layers[i]
            is_sliding = layer_types[i] == 'sliding_attention'
            hd = head_dim if is_sliding else global_head_dim
            rope = local_rope if is_sliding else global_rope
            residual = x
            x = rms_norm(x, lp['input_layernorm'], rms_norm_eps)
            q = (x @ lp['self_attn.q_proj'].T).reshape(bs, seq_len, heads, hd)
            q = rms_norm(q, lp['self_attn.q_norm'], rms_norm_eps)
            q = apply_rope(q, rope[0], rope[1], position_ids)
            if i < first_kv_shared_layer:
                k = (x @ lp['self_attn.k_proj'].T).reshape(bs, seq_len, kv_heads, hd)
                v = (x @ lp['self_attn.v_proj'].T).reshape(bs, seq_len, kv_heads, hd)
                k = rms_norm(k, lp['self_attn.k_norm'], rms_norm_eps)
                v = rms_norm(v, eps=rms_norm_eps)
                k = apply_rope(k, rope[0], rope[1], position_ids)
                k_cache = jax.lax.dynamic_update_slice(kv_cache[i][0], k.astype(jnp.bfloat16), (0, pos, 0, 0))
                v_cache = jax.lax.dynamic_update_slice(kv_cache[i][1], v.astype(jnp.bfloat16), (0, pos, 0, 0))
                new_cache.append((k_cache, v_cache))
            else:
                donor = kv_donor_sliding if is_sliding else kv_donor_full
                k_cache = new_cache[donor][0]
                v_cache = new_cache[donor][1]
                new_cache.append(kv_cache[i])
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
            attn_out = attention(q, k_rep, v_rep, mask).reshape(bs, seq_len, heads * hd)
            attn_out = attn_out @ lp['self_attn.o_proj'].T
            x = residual + rms_norm(attn_out, lp['post_attention_layernorm'], rms_norm_eps)
            residual = x
            x = rms_norm(x, lp['pre_feedforward_layernorm'], rms_norm_eps)
            gate = x @ lp['mlp.gate_proj'].T
            up = x @ lp['mlp.up_proj'].T
            x = jax.nn.gelu(gate, approximate=True) * up
            x = x @ lp['mlp.down_proj'].T
            x = residual + rms_norm(x, lp['post_feedforward_layernorm'], rms_norm_eps)
            residual = x
            gate_ple = x @ lp['per_layer_input_gate'].T
            gate_ple = jax.nn.gelu(gate_ple, approximate=True)
            gate_ple = gate_ple * per_layer_inputs[:, :, i, :]
            projected = gate_ple @ lp['per_layer_projection'].T
            projected = rms_norm(projected, lp['post_per_layer_input_norm'], rms_norm_eps)
            x = residual + projected
            x = x * lp['layer_scalar']
        x = rms_norm(x, params[f'{lm}.norm.weight'], rms_norm_eps)
        logits = (x @ embed.T)[:, -1, :]
        if final_logit_softcapping is not None:
            logits = jnp.tanh(logits / final_logit_softcapping) * final_logit_softcapping
        return logits, new_cache

    def forward(self, input_ids, kv_cache, pos, temperature=0.7, top_k=40):
        if kv_cache is None:
            batch_size = input_ids.shape[0]
            kv_cache = []
            for i in range(self.num_layers):
                hd = self.head_dim if self.layer_types[i] == 'sliding_attention' else self.global_head_dim
                shape = (batch_size, self.max_cache_len, self.kv_heads, hd)
                kv_cache.append((jnp.zeros(shape, jnp.bfloat16), jnp.zeros(shape, jnp.bfloat16)))
        logits, kv_cache = Gemma4.forward_jit(
            input_ids, self.params, self.layers, self.embed, self.embed_per_layer,
            self.local_rope, self.global_rope, kv_cache, pos,
            self.num_layers, self.heads, self.kv_heads, self.head_dim,
            self.global_head_dim, self.hidden, self.rms_norm_eps,
            self.layer_types, self.sliding_window, self.max_cache_len,
            self.first_kv_shared_layer, self.kv_donor_sliding, self.kv_donor_full,
            self.ple_dim, self.final_logit_softcapping)
        logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            vals, _ = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
            logits = jnp.where(logits >= vals[:, -1:], logits, jnp.finfo(logits.dtype).min)
        self.rng_key, key = jax.random.split(self.rng_key)
        return (int(jnp.argmax(logits[0])) if temperature == 0 else int(jax.random.categorical(key, logits)[0])), kv_cache


prompt = 'What is the capital of France?' if len(sys.argv) <= 1 else sys.argv[1]
model_type = 'google/gemma-4-E2B-it'
tokenizer = download(model_type, 'tokenizer.json')
config = download(model_type, 'config.json')
params = download(model_type, 'model.safetensors')
tokenizer = Tokenizer(tokenizer)
with open(config) as file:
    config = json.load(file)
params = load_safetensors(params)
model = Gemma4(config, params)
input_ids = jnp.array([tokenizer.encode(f'<|turn>user\n{prompt}<turn|>\n<|turn>model\n', add_bos=True)], dtype=jnp.int32)
kv_cache = None
pos = 0
for _ in range(model.max_cache_len):
    token, kv_cache = model.forward(input_ids, kv_cache, pos, temperature=0.0)
    pos += input_ids.shape[1]
    if token in tokenizer.eos_ids:
        break
    print(tokenizer.decode([token]), end='', flush=True)
    input_ids = jnp.array([[token]], dtype=jnp.int32)
print()
