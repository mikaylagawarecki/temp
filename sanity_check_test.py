import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.autograd import gradcheck

from mha import MultiHeadAttention
from te_layer import TransformerEncoderLayer
from td_layer import TransformerDecoderLayer
from utils import gen_batch, jagged_to_padded, benchmark

class TestMHA(TestCase):

    def _set_seeds(self):
        torch.manual_seed(5)
        np.random.seed(0)
        torch._dynamo.reset_code_caches()

    @parametrize("bias", [True, False])
    @parametrize("kdim", [512, 128])
    @parametrize("vdim", [512, 128])
    @parametrize("device", ["cuda"])
    def test_mha_parity(self, bias, kdim, vdim, device):
        N, E_q, E_k, E_v, E_total = 512, 512, kdim, vdim, 512
        batch_first = True  # NJT does not support seq_len as first dim
        d_model = E_q
        nheads = 8
        dropout = 0.0
        qkv_same_embed_dim = E_q == E_k and E_q == E_v

        self._set_seeds()
        vanilla_mha_layer = nn.MultiheadAttention(E_q,
                                                  nheads,
                                                  dropout=dropout,
                                                  bias=bias,
                                                  batch_first=batch_first,
                                                  kdim=kdim,
                                                  vdim=vdim,
                                                  device=device)
        compiled_vanilla_mha_layer = torch.compile(vanilla_mha_layer)

        self._set_seeds()
        mha_layer = MultiHeadAttention(E_q,
                                       E_k,
                                       E_v,
                                       E_total,
                                       nheads,
                                       dropout=dropout,
                                       bias=bias,
                                       device=device)
        compiled_mha_layer = torch.compile(mha_layer)
    
        # nn.MultiheadAttention uses a non conventional init for linear weights/biases, so do this :(
        mha_layer.out_proj.weight = nn.Parameter(vanilla_mha_layer.out_proj.weight.clone().detach())
        if bias:
            mha_layer.out_proj.bias = nn.Parameter(vanilla_mha_layer.out_proj.bias.clone().detach())
        if qkv_same_embed_dim:
            mha_layer.packed_proj.weight = nn.Parameter(vanilla_mha_layer.in_proj_weight.clone().detach())
            if bias:
                mha_layer.packed_proj.bias = nn.Parameter(vanilla_mha_layer.in_proj_bias.clone().detach())
        else:
            mha_layer.q_proj.weight = nn.Parameter(vanilla_mha_layer.q_proj_weight.clone().detach())
            mha_layer.k_proj.weight = nn.Parameter(vanilla_mha_layer.k_proj_weight.clone().detach())
            mha_layer.v_proj.weight = nn.Parameter(vanilla_mha_layer.v_proj_weight.clone().detach())
            if bias:
                mha_layer.q_proj.bias = nn.Parameter(vanilla_mha_layer.in_proj_bias[:E_total].clone().detach())
                mha_layer.k_proj.bias = nn.Parameter(vanilla_mha_layer.in_proj_bias[E_total:2*E_total].clone().detach())
                mha_layer.v_proj.bias = nn.Parameter(vanilla_mha_layer.in_proj_bias[2*E_total:].clone().detach())


        self._set_seeds()
        query, key, value, sentence_lengths = gen_batch(N, E_q, E_k, E_v, device=device)
        padded_query, padded_key, padded_value = query.to_padded_tensor(0.0), key.to_padded_tensor(0.0), value.to_padded_tensor(0.0)

        max_seq_len = sentence_lengths.max().item()

        key_padding_mask = torch.where(padded_key == 0.0, float('-inf'), 0)[:, :, 0]
        attn_mask = torch.empty((N, max_seq_len, max_seq_len), device=device).fill_(float('-inf'))
        for i, s in enumerate(sentence_lengths):
            attn_mask[i, :s, :s] = nn.Transformer.generate_square_subsequent_mask(s)
        attn_mask = attn_mask.unsqueeze(1).expand(N, nheads, max_seq_len, max_seq_len).reshape(N*nheads, max_seq_len, max_seq_len)

        # warmup
        compiled_vanilla_mha_layer(padded_query,
                                   padded_key,
                                   padded_value,
                                   key_padding_mask=key_padding_mask,
                                   attn_mask=attn_mask,
                                   need_weights=False,
                                   is_causal=True)
        compiled_mha_layer(query, key, value, is_causal=True)

        # benchmark
        (vanilla_result, _), vanilla_time = benchmark(compiled_vanilla_mha_layer,
                                                      padded_query,
                                                      padded_key,
                                                      padded_value,
                                                      key_padding_mask=key_padding_mask,
                                                      attn_mask=attn_mask,
                                                      need_weights=False)
        mha_result, mha_time = benchmark(compiled_mha_layer, query, key, value, is_causal=True)
        padded_mha_result = mha_result.to_padded_tensor(0.0)

        # padding-specific step: remove output projection bias from padded entries for fair comparison
        if bias:
            for i, entry_length in enumerate(sentence_lengths):
                vanilla_result[i, entry_length:, :] = 0.0

        self.assertEqual(vanilla_result, padded_mha_result)
        self.assertTrue(vanilla_time > mha_time)

        vanilla_result.sum().backward()
        padded_mha_result.sum().backward()

        atol, rtol = 1e-3, 1e-3
        self.assertEqual(mha_layer.out_proj.weight.grad, vanilla_mha_layer.out_proj.weight.grad, atol=atol, rtol=rtol)
        if bias:
            self.assertEqual(mha_layer.out_proj.bias.grad, vanilla_mha_layer.out_proj.bias.grad, atol=atol, rtol=rtol)
        if qkv_same_embed_dim:
            self.assertEqual(mha_layer.packed_proj.weight.grad, vanilla_mha_layer.in_proj_weight.grad, atol=atol, rtol=rtol)
            if bias:
                self.assertEqual(mha_layer.packed_proj.bias.grad, vanilla_mha_layer.in_proj_bias.grad, atol=atol, rtol=rtol)
        else:
            self.assertEqual(mha_layer.q_proj.weight.grad, vanilla_mha_layer.q_proj_weight.grad, atol=atol, rtol=rtol)
            self.assertEqual(mha_layer.k_proj.weight.grad, vanilla_mha_layer.k_proj_weight.grad, atol=atol, rtol=rtol)
            self.assertEqual(mha_layer.v_proj.weight.grad, vanilla_mha_layer.v_proj_weight.grad, atol=atol, rtol=rtol)
            if bias:
                self.assertEqual(mha_layer.q_proj.bias.grad, vanilla_mha_layer.in_proj_bias.grad[:E_total], atol=atol, rtol=rtol)
                self.assertEqual(mha_layer.k_proj.bias.grad, vanilla_mha_layer.in_proj_bias.grad[E_total:2*E_total], atol=atol, rtol=rtol)
                self.assertEqual(mha_layer.v_proj.bias.grad, vanilla_mha_layer.in_proj_bias.grad[2*E_total:], atol=atol, rtol=rtol)         


    @parametrize("activation", [F.relu, F.gelu])
    @parametrize("norm_first", [True, False])
    @parametrize("bias", [True, False])
    @parametrize("device", ["cuda"])
    def test_te_layer_parity(self, activation, norm_first, bias, device):
        N, E_q, E_k, E_v, E_total = 512, 512, 512, 512, 512
        d_model = E_q
        nheads = 8
        dropout = 0.0
        batch_first = True # NJT does not support seq_len as first dim

        self._set_seeds()
        vanilla_te_layer = nn.TransformerEncoderLayer(d_model,
                                                      nheads,
                                                      dropout=dropout,
                                                      activation=activation,
                                                      batch_first=batch_first,
                                                      norm_first=norm_first,
                                                      bias=bias,
                                                      device=device)
        compiled_vanilla_te_layer = torch.compile(vanilla_te_layer)

        self._set_seeds()
        te_layer = TransformerEncoderLayer(d_model,
                                           nheads,
                                           dropout=dropout,
                                           activation=activation,
                                           norm_first=norm_first,
                                           bias=bias,
                                           device=device)
        compiled_te_layer = torch.compile(te_layer)
    
        # nn.MultiheadAttention uses a non conventional init for linear weights, so do this :(
        te_layer.self_attn.out_proj.weight = nn.Parameter(vanilla_te_layer.self_attn.out_proj.weight.clone().detach())
        te_layer.self_attn.packed_proj.weight = nn.Parameter(vanilla_te_layer.self_attn.in_proj_weight.clone().detach())
        if bias:
            # Turning bias on changes the random seeds for everything, so do this
            te_layer.linear1.weight = nn.Parameter(vanilla_te_layer.linear1.weight.clone().detach())
            te_layer.linear2.weight = nn.Parameter(vanilla_te_layer.linear2.weight.clone().detach())
            te_layer.norm1.weight = nn.Parameter(vanilla_te_layer.norm1.weight.clone().detach())
            te_layer.norm2.weight = nn.Parameter(vanilla_te_layer.norm2.weight.clone().detach())
            te_layer.linear1.bias = nn.Parameter(vanilla_te_layer.linear1.bias.clone().detach())
            te_layer.linear2.bias = nn.Parameter(vanilla_te_layer.linear2.bias.clone().detach())
            te_layer.norm1.bias = nn.Parameter(vanilla_te_layer.norm1.bias.clone().detach())
            te_layer.norm2.bias = nn.Parameter(vanilla_te_layer.norm2.bias.clone().detach())
            te_layer.self_attn.out_proj.bias = nn.Parameter(vanilla_te_layer.self_attn.out_proj.bias.clone().detach())
            te_layer.self_attn.packed_proj.bias = nn.Parameter(vanilla_te_layer.self_attn.in_proj_bias.clone().detach())

        self._set_seeds()
        query, _, _, sentence_lengths = gen_batch(N, E_q, E_k, E_v, device=device)
        padded_query = query.to_padded_tensor(0.0)

        max_seq_len = sentence_lengths.max().item()

        key_padding_mask = torch.where(padded_query == 0.0, -math.inf, 0)[:, :, 0]
        attn_mask = torch.empty((N, max_seq_len, max_seq_len), device=device).fill_(float('-inf'))
        for i, s in enumerate(sentence_lengths):
            attn_mask[i, :s, :s] = nn.Transformer.generate_square_subsequent_mask(s)
        attn_mask = attn_mask.unsqueeze(1).expand(N, nheads, max_seq_len, max_seq_len).reshape(N*nheads, max_seq_len, max_seq_len)

        # warmup
        compiled_vanilla_te_layer(padded_query,
                                  src_key_padding_mask=key_padding_mask,
                                  src_mask=attn_mask,
                                  is_causal=True)
        compiled_te_layer(query, is_causal=True)

        # benchmark
        vanilla_result, vanilla_time = benchmark(compiled_vanilla_te_layer,
                                                 padded_query,
                                                 src_key_padding_mask=key_padding_mask,
                                                 src_mask=attn_mask,
                                                 is_causal=True)
        te_result, te_time = benchmark(compiled_te_layer, query, is_causal=True)
        padded_te_result = te_result.to_padded_tensor(0.0)

        # padding-specific step: remove output projection bias from padded entries for fair comparison
        if bias:
            for i, entry_length in enumerate(sentence_lengths):
                vanilla_result[i, entry_length:, :] = 0.0

        self.assertEqual(vanilla_result, padded_te_result, atol=1e-2, rtol=1e-2)
        self.assertTrue(vanilla_time > te_time)

        vanilla_result.sum().backward()
        padded_te_result.sum().backward()

        # FIXME: are these atol and rtol okay
        # self.assertEqual(te_layer.out_proj.weight.grad, vanilla_te_layer.out_proj.weight.grad, atol=1e-3, rtol=1e-3)
        # self.assertEqual(te_layer.packed_proj.weight.grad, vanilla_te_layer.in_proj_weight.grad, atol=1e-3, rtol=1e-3)
        # Bias gradients seem hugely wrong?
        # if bias:
        #     self.assertEqual(mha_layer.out_proj.bias.grad, vanilla_mha_layer.out_proj.bias.grad, atol=1e-3, rtol=1e-3)
        #     self.assertEqual(mha_layer.packed_proj.bias.grad, vanilla_mha_layer.in_proj_bias.grad, atol=1e-3, rtol=1e-3)


    @parametrize("activation", [F.relu, F.gelu])
    @parametrize("bias", [True, False])
    @parametrize("norm_first", [True, False])
    @parametrize("device", ["cuda"])
    def test_td_layer_parity(self, activation, bias, norm_first, device):
        N, E_q, E_k, E_v, E_total = 512, 512, 512, 512, 512
        d_model = E_q
        nheads = 8
        dropout = 0.0
        batch_first = True # NJT does not support seq_len as first dim

        self._set_seeds()
        vanilla_td_layer = nn.TransformerDecoderLayer(d_model,
                                                      nheads,
                                                      dropout=dropout,
                                                      activation=activation,
                                                      batch_first=batch_first,
                                                      norm_first=norm_first,
                                                      bias=bias,
                                                      device=device)
        compiled_vanilla_td_layer = torch.compile(vanilla_td_layer)

        self._set_seeds()
        td_layer = TransformerDecoderLayer(d_model,
                                           nheads,
                                           dropout=dropout,
                                           activation=activation,
                                           norm_first=norm_first,
                                           bias=bias,
                                           device=device)
        compiled_td_layer = torch.compile(td_layer)

        # Query from different sequence than key/value
        query, _, _, q_seq_len = gen_batch(N, E_q, E_k, E_v, device=device)
        _, memory, _, kv_seq_len = gen_batch(N, E_q, E_k, E_v, device=device)
        padded_query, padded_memory = query.to_padded_tensor(0.0), memory.to_padded_tensor(0.0)

        # nn.MultiheadAttention uses a non conventional init for linear weights, so do this :(
        td_layer.self_attn.out_proj.weight = nn.Parameter(vanilla_td_layer.self_attn.out_proj.weight.clone().detach())
        td_layer.self_attn.packed_proj.weight = nn.Parameter(vanilla_td_layer.self_attn.in_proj_weight.clone().detach())
        td_layer.multihead_attn.out_proj.weight = nn.Parameter(vanilla_td_layer.multihead_attn.out_proj.weight.clone().detach())
        td_layer.multihead_attn.packed_proj.weight = nn.Parameter(vanilla_td_layer.multihead_attn.in_proj_weight.clone().detach())
        
        if bias:
            # Turning bias on changes the random seeds for everything, so do this
            td_layer.linear1.weight = nn.Parameter(vanilla_td_layer.linear1.weight.clone().detach())
            td_layer.linear2.weight = nn.Parameter(vanilla_td_layer.linear2.weight.clone().detach())
            td_layer.norm1.weight = nn.Parameter(vanilla_td_layer.norm1.weight.clone().detach())
            td_layer.norm2.weight = nn.Parameter(vanilla_td_layer.norm2.weight.clone().detach())
            td_layer.norm3.weight = nn.Parameter(vanilla_td_layer.norm3.weight.clone().detach())
            td_layer.linear1.bias = nn.Parameter(vanilla_td_layer.linear1.bias.clone().detach())
            td_layer.linear2.bias = nn.Parameter(vanilla_td_layer.linear2.bias.clone().detach())
            td_layer.norm1.bias = nn.Parameter(vanilla_td_layer.norm1.bias.clone().detach())
            td_layer.norm2.bias = nn.Parameter(vanilla_td_layer.norm2.bias.clone().detach())
            td_layer.norm3.bias = nn.Parameter(vanilla_td_layer.norm3.bias.clone().detach())
            td_layer.self_attn.out_proj.bias = nn.Parameter(vanilla_td_layer.self_attn.out_proj.bias.clone().detach())
            td_layer.self_attn.packed_proj.bias = nn.Parameter(vanilla_td_layer.self_attn.in_proj_bias.clone().detach())
            td_layer.multihead_attn.out_proj.bias = nn.Parameter(vanilla_td_layer.multihead_attn.out_proj.bias.clone().detach())
            td_layer.multihead_attn.packed_proj.bias = nn.Parameter(vanilla_td_layer.multihead_attn.in_proj_bias.clone().detach())

    
        for (n1, p1), (n2, p2) in zip(td_layer.named_parameters(), vanilla_td_layer.named_parameters()):
            self.assertEqual(p1, p2)

        # Create the masks
        tgt_key_padding_mask = torch.where(padded_query == 0.0, -math.inf, 0)[:, :, 0]
        memory_key_padding_mask = torch.where(padded_memory == 0.0, -math.inf, 0)[:, :, 0]
        q_max_seq_len, kv_max_seq_len = q_seq_len.max().item(), kv_seq_len.max().item()
        tgt_mask = torch.empty((N, q_max_seq_len, q_max_seq_len), device=device).fill_(float('-inf'))
        memory_mask = torch.empty((N, q_max_seq_len, kv_max_seq_len), device=device).fill_(float('-inf'))
        for i, s in enumerate(q_seq_len):
            tgt_mask[i, :s, :s] = nn.Transformer.generate_square_subsequent_mask(s)
        for i, (s1, s2) in enumerate(zip(q_seq_len, kv_seq_len)):
            memory_mask[i, :s1, :s2] = torch.where(torch.tril(torch.ones((s1, s2), device=device)) == 0, float('-inf'), 0)

        tgt_mask = tgt_mask.unsqueeze(1).expand(N, nheads, q_max_seq_len, q_max_seq_len).reshape(N*nheads, q_max_seq_len, q_max_seq_len)
        memory_mask = memory_mask.unsqueeze(1).expand(N, nheads, q_max_seq_len, kv_max_seq_len).reshape(N*nheads, q_max_seq_len, kv_max_seq_len)
        
        # warmup
        compiled_vanilla_td_layer(padded_query,
                                  padded_memory,
                                  tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  tgt_is_causal=True,
                                  memory_is_causal=True)
        compiled_td_layer(query, memory, tgt_is_causal=True, memory_is_causal=True)

        # benchmark
        vanilla_result, vanilla_time = benchmark(compiled_vanilla_td_layer,
                                                 padded_query,
                                                 padded_memory,
                                                 tgt_mask=tgt_mask,
                                                 memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask,
                                                 tgt_is_causal=True,
                                                 memory_is_causal=True)
        td_result, td_time = benchmark(compiled_td_layer, query, memory, tgt_is_causal=True, memory_is_causal=True)
        padded_td_result = td_result.to_padded_tensor(0.0)

        # padding-specific step: remove output projection bias from padded entries for fair comparison
        if bias:
            for i, entry_length in enumerate(q_seq_len):
                vanilla_result[i, entry_length:, :] = 0.0
        
        self.assertEqual(vanilla_result, padded_td_result, atol=1e-3, rtol=1e-3)
        self.assertTrue(vanilla_time > td_time)

    
instantiate_parametrized_tests(TestMHA)

if __name__ == '__main__':
    run_tests()
