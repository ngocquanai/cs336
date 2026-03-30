
from cs336_basics.tokenizer import Tokenizer, tokenize_data
import numpy as np
import random
import torch

text_file = "../data/test.txt"
vocab_path = "../data/TinyStories_vocab.json"
merges_path = "../data/TinyStories_merges.txt"
saved_path = "../data/token_ids.npy"
special_tokens = ["<|endoftext|>"]



def text_to_ids(text_file, vocab_file, merges_file, saved_path, special_tokens= ["<|endoftext|>"], num_processes= 128) :
    tokenizer = Tokenizer.from_files(vocab_file, merges_file, special_tokens= special_tokens)
    token_ids = tokenize_data(tokenizer, filepath= text_file, special_token= special_tokens[0], num_processes= num_processes)
    token_ids = np.array(token_ids)
    np.save(saved_path, token_ids)
    print(f"Successful saved to {saved_path}")

# text_to_ids(text_file, vocab_path, merges_path, saved_path, special_tokens= special_tokens, num_processes= 128)


def data_loading(dataset, batch_size, context_len, device) :
    if len(dataset) < context_len :
        raise ValueError(f"Data is too small, length smaller than context len {context_len}")

    valid_idx = len(dataset) - context_len - 1

    start_indices = [random.randint(0, valid_idx) for _ in range(batch_size)]

    token_ids_list = [dataset[start_idx:start_idx+context_len+1] for start_idx in start_indices]
    x_token_ids = [token_ids[:-1] for token_ids in token_ids_list]
    y_token_ids = [token_ids[1:] for token_ids in token_ids_list]

    x_token_ids = torch.tensor(np.array(x_token_ids), device= device, dtype= torch.long)
    y_token_ids = torch.tensor(np.array(y_token_ids), device= device, dtype= torch.long)

    return x_token_ids, y_token_ids


