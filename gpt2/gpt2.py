import collections
import json
import math
import os
import re
import sys
import urllib.request

import torch


def download(url, file=None):
    file = file if file else url.split('/')[-1]
    path = os.path.join(os.path.dirname(__file__), file)
    if not os.path.isfile(path):
        def reporthook(count, block_size, total_size):
            percent = str(int(100 * count * block_size / total_size)) + '%'
            print(f"\r\033[KDownloading '{file}' ({percent})", end='', flush=True)
        urllib.request.urlretrieve(url, path, reporthook=reporthook)
        print('\r\033[K', end='', flush=True)
    return path

class BPETokenizer:

    def __init__(self):
        url = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/'
        with open(download(f"{url}encoder.json"), encoding='utf-8') as file:
            self.encoder = json.load(file)
        self.decoder = {v:k for k,v in self.encoder.items()}
        with open(download(f"{url}vocab.bpe"), encoding='utf-8') as file:
            vocab = file.read().split('\n')[1:-1]
        self.bpe_ranks = {tuple(line.split()): i for i, line in enumerate(vocab)}
        assert len(self.encoder) == 50257 and len(self.bpe_ranks) == 50000
        bs = list(range(33, 127)) + list(range(161, 256))
        xs = list(range(0, 33)) + list(range(127, 161))
        cs = bs[:] + [2**8 + i for i in range(len(xs))]
        self.byte_encoder = dict(zip(bs + xs, [chr(n) for n in cs]))
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}

    def encode(self, text, allowed_special=None):
        tokens = re.findall(r"""<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d| ?""" +
                            r"""[A-Za-z_]+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""", text, re.UNICODE)
        def translate(token):
            if token == '<|endoftext|>':
                assert allowed_special and token in allowed_special
                return [token]
            word = tuple(''.join(self.byte_encoder[byte] for byte in token.encode('utf-8')))
            while len(word) != 1:
                pairs = set((word[i], word[i+1]) for i in range(len(word)-1))
                bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
                if bigram not in self.bpe_ranks:
                    break
                a, b = bigram
                new_word = []
                i = 0
                while i < len(word):
                    j = word.index(a, i) if a in word[i:] else len(word)
                    new_word.extend(word[i:j])
                    i = j
                    if i < len(word):
                        j = 2 if i < len(word)-1 and word[i] == a and word[i+1] == b else 1
                        new_word.append(a+b if j == 2 else word[i])
                        i += j
                word = tuple(new_word)
            return word
        return [self.encoder[_] for token in tokens for _ in translate(token)]

    def decode(self, tokens):
        tokens = [self.decoder[token] for token in tokens]
        buffer = bytearray([self.byte_decoder[c] for c in ''.join(tokens)])
        return buffer.decode('utf-8', errors='replace')

class GPT2Config:

    def __init__(self, model_type):
        configs = {
            'gpt2':        [ 12, 12, 768  ], # 124M params
            'gpt2-medium': [ 24, 16, 1024 ], # 350M params
            'gpt2-large':  [ 36, 20, 1280 ], # 774M params
            'gpt2-xl':     [ 48, 25, 1600 ]  # 1558M params
        }
        self.n_layer, self.n_head, self.n_embd = configs[model_type]
        self.type = model_type
        self.vocab_size = 50257
        self.block_size = 1024
        self.url = f'https://huggingface.co/{model_type}/resolve/main/pytorch_model.bin'

class Attention(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        size = config.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.mlp = torch.nn.Sequential(collections.OrderedDict([
            ('c_fc', torch.nn.Linear(config.n_embd, 4 * config.n_embd)),
            ('act', torch.nn.GELU(approximate='tanh')),
            ('c_proj', torch.nn.Linear(4 * config.n_embd, config.n_embd))
        ]))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(torch.nn.Module):

    def __init__(self, model_type):
        super().__init__()
        config = GPT2Config(model_type)
        self.block_size = config.block_size
        self.transformer = torch.nn.ModuleDict({
            'wte': torch.nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': torch.nn.Embedding(config.block_size, config.n_embd),
            'h': torch.nn.Sequential(*[Block(config) for _ in range(config.n_layer)]),
            'ln_f': torch.nn.LayerNorm(config.n_embd)
        })
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        state_file = download(config.url, config.type + '.bin')
        state_dict = torch.load(state_file, weights_only=True)
        transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
        for key, value in state_dict.items():
            if any(key.endswith(w) for w in transposed):
                state_dict[key] = value.t()
        self.transformer.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor, temperature: float = 0.1, top_k: int = 40):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        x = self.lm_head(self.transformer.ln_f(self.transformer.h(x)))
        logits = torch.select(x, dim=1, index=-1) / temperature
        min_top_k = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = torch.where(logits >= min_top_k, logits, -float('Inf'))
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

prompt = 'The Eiffel tower is in' if len(sys.argv) <= 1 else sys.argv[1]
model = GPT2('gpt2').eval()
encoding = BPETokenizer()
# import tiktoken
# encoding = tiktoken.get_encoding('gpt2')
prompt = prompt if prompt != '' else '<|endoftext|>'
context = encoding.encode(prompt, allowed_special={'<|endoftext|>'})
print(encoding.decode(context), end='', flush=True)
context = torch.tensor([context], dtype=torch.long)
for _ in range(50): # max_tokens
    y = model(context)
    context = torch.cat((context, y), dim=1)
    w = encoding.decode(y[0].tolist())
    print(w, end='', flush=True)
print('')
