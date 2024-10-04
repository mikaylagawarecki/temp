import math
import numpy as np
import timeit
import torch

# ==== Utilities to generate data ====

def zipf_sentence_lengths(alpha: float, batch_size: int) -> torch.Tensor:
    # generate fake corpus by unigram Zipf distribution
    # from wikitext-2 corpus, we get rank "." = 3, "!" = 386, "?" = 858
    sentence_lengths = np.empty(batch_size, dtype=int)
    for ibatch in range(batch_size):
        sentence_lengths[ibatch] = 1
        word = np.random.zipf(alpha)
        while word != 3 and word != 386 and word != 858:
            sentence_lengths[ibatch] += 1
            word = np.random.zipf(alpha)
    return torch.tensor(sentence_lengths)

def gen_batch(N, E_q, E_k, E_v, device, dtype=torch.float32, query_seq_len_1=False):
    # generate semi-realistic data using Zipf distribution for sentence lengths
    sentence_lengths = zipf_sentence_lengths(alpha=1.2, batch_size=N)

    # Note: the torch.jagged layout is a nested tensor layout that supports a single ragged
    # dimension and works with torch.compile. The batch items each have shape (B, S*, D)
    # where B = batch size, S* = ragged sequence length, and D = embedding dimension.
    if query_seq_len_1:
      query = torch.nested.nested_tensor([
          torch.randn(1, E_q, device=device, dtype=dtype)
          for l in sentence_lengths
      ], layout=torch.jagged)
    else:
      query = torch.nested.nested_tensor([
          torch.randn(l.item(), E_q, device=device, dtype=dtype)
          for l in sentence_lengths
      ], layout=torch.jagged)

    key = torch.nested.nested_tensor([
        torch.randn(s.item(), E_k, device=device, dtype=dtype)
        for s in sentence_lengths
    ], layout=torch.jagged)

    value = torch.nested.nested_tensor([
        torch.randn(s.item(), E_v, device=device, dtype=dtype)
        for s in sentence_lengths
    ], layout=torch.jagged)

    return query, key, value, sentence_lengths

# FIXME: can remove this one
def jagged_to_padded(jt, padding_val):
    # TODO: do jagged -> padded directly when this is supported
    return torch.nested.to_padded_tensor(
        torch.nested.nested_tensor(list(jt.unbind())),
        padding_val)


def benchmark(func, *args, **kwargs):
    torch.cuda.synchronize()
    begin = timeit.default_timer()
    output = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    return output, (end - begin)
