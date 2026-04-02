from cs336_basics.pretokenization import pretokenization
import os
from tqdm import tqdm
import time
import json
import base64

from cs336_basics.utils.constant import GPT2_PRETOKENIZATION
from cs336_basics.utils.io import save_vocab_merges, load_vocab_merges


def init_vocab(special_tokens: list[str]) -> dict :
    vocab = { i : bytes([i]) for i in range(256)}

    # add special tokens
    for idx in range(len(special_tokens)) :
        token = special_tokens[idx]
        vocab[256 + idx] = token.encode('utf-8')
    return vocab

def most_frequent_pair(pairs_count) :
    item = max(pairs_count.items(), key= lambda x: (x[1], x[0]))
    return item[0]

def _merge_bytes(byte_tuple, selected_pair):

    new_byte = selected_pair[0] + selected_pair[1]
    result = []
    idx = 0
    while idx < len(byte_tuple):
        if idx < len(byte_tuple) - 1 and (byte_tuple[idx], byte_tuple[idx + 1]) == selected_pair:
            result.append(new_byte)
            idx += 2 
        else:
            result.append(byte_tuple[idx])
            idx += 1
    return tuple(result)


def count_adjacent_pair(freq_table: dict) :

    pairs_count = dict()
    pairs_to_tuples = dict() 

    for byte_tuple, count in tqdm(freq_table.items()) :
        for byte1, byte2 in zip(byte_tuple[:-1], byte_tuple[1:]) :
            pair = (byte1, byte2)
            pairs_count[pair] = pairs_count.get(pair, 0) + count

            if pair not in pairs_to_tuples :
                pairs_to_tuples[pair] = set()
            pairs_to_tuples[pair].add(byte_tuple)

    return pairs_count, pairs_to_tuples

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_processes: int = 1) :
    start_time = time.time()
    PAT = GPT2_PRETOKENIZATION
    vocab = init_vocab(special_tokens)
    merges = []
    freq_table = pretokenization(
        filepath= input_path,
        PAT= PAT,
        special_tokens= special_tokens,
        num_processes= num_processes
    )
    end_time = time.time()
    print(f"Pretokenization time: {int((end_time - start_time)*1000)/1000}s")
    pairs_count, pairs_to_tuples = count_adjacent_pair(freq_table)



    origin_len = len(vocab)
    pbar = tqdm(total= vocab_size - origin_len)
    while len(vocab) < vocab_size :
        if not pairs_count :
            break
        selected_pair = most_frequent_pair(pairs_count)
        new_byte = selected_pair[0] + selected_pair[1]

        # update merges
        merges.append(selected_pair)

        # Update vocab
        new_vocab_idx = len(vocab)
        vocab[new_vocab_idx] = new_byte

        affected_tuples = list(pairs_to_tuples.get(selected_pair, set()))  

        for original_byte_tuple in affected_tuples :
            count = freq_table[original_byte_tuple]
            byte_tuple = _merge_bytes(original_byte_tuple, selected_pair)

            # Update frequency table
            if byte_tuple != original_byte_tuple :
                freq_table[byte_tuple] = count
                del freq_table[original_byte_tuple]


            # Update pairs_to_tuples & pairs_count
            for byte1, byte2 in zip(original_byte_tuple[:-1], original_byte_tuple[1:]) :
                old_pair = (byte1, byte2)
                pairs_to_tuples[old_pair].discard(original_byte_tuple)
                pairs_count[old_pair] -= count
            for byte1, byte2 in zip(byte_tuple[:-1], byte_tuple[1:]) :
                new_pair = (byte1, byte2)
                if new_pair not in pairs_to_tuples:  
                    pairs_to_tuples[new_pair] = set()
                pairs_to_tuples[new_pair].add(byte_tuple)
                pairs_count[new_pair] = pairs_count.get(new_pair, 0) + count


        pbar.update(len(vocab) - origin_len - pbar.n)
    pbar.close()
    # breakpoint()

    return vocab, merges










                












