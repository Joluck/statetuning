########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from .block import Block

class RWKV7(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        兼容 transformers 的 generate() 接口.
        对 RWKV 来说，我们不需要做实际处理，直接返回原始输入即可。
        """
        return {"input_ids": input_ids, **kwargs}

    def get_input_embeddings(self):
        """为 PEFT 提供 Embedding 层引用"""
        return self.emb

    def set_input_embeddings(self, new_emb):
        """允许 PEFT 替换 Embedding 层（通常不会触发）"""
        self.emb = new_emb

    def get_output_embeddings(self):
        """为 PEFT 提供输出 head 层引用"""
        return self.head

    def set_output_embeddings(self, new_head):
        """允许 PEFT 替换输出 head 层"""
        self.head = new_head
    
    def forward(self, input_ids):
        args = self.args
        B, T = input_ids.size()

        x = self.emb(input_ids)
        v_first = torch.empty_like(x)

        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)

        return x

