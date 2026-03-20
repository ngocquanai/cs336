from cs336_basics.pretokenization import pretokenization
import os

def init_vocab(special_tokens) -> dict :
    vocab = { i : bytes([i]) for i in range(256)}

    # add special tokens
    for idx in range(len(special_tokens)) :
        token = special_tokens[idx]
        vocab[256 + idx] = token.encode('utf-8')
    return vocab

def most_frequent_pair(pairs_count) :
    item = max(pairs_count.items(), key= lambda x: (x[1], x[0]))

    return item[0]

def merge_bytes(byte_tuple, pair) :
    result = []
    i = 0
    while i < len(byte_tuple) :
        if i < len(byte_tuple) - 1 and (byte_tuple[i], byte_tuple[i+1]) == pair :
            # merge pair
            result.append(byte_tuple[i] + byte_tuple[i+1])
            i += 2
        else :
            result.append(byte_tuple[i])
            i+= 1
    

    return tuple(result)

def adjacent_pair(freq_table: dict) :

    pairs_count = dict()

    for word, count in freq_table.items() :
        for byte1, byte2 in zip(word[:-1], word[1:]) :
            pair = (byte1, byte2)
            pairs_count[pair] = pairs_count.get(pair, 0) + count
    return pairs_count


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_processes= None) :

    if not num_processes :
        num_processes = 128
        print(num_processes, "!"*200)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab = init_vocab(special_tokens)
    merges = []
    freq_table = pretokenization(
        filepath= input_path,
        PAT= PAT,
        special_tokens= special_tokens,
        num_processes= num_processes
    )
    pairs_count = adjacent_pair(freq_table)

    while len(vocab) < vocab_size :
        pairs_count = adjacent_pair(freq_table)
        if not pairs_count :
            break
        selected_pair = most_frequent_pair(pairs_count)
        new_byte = selected_pair[0] + selected_pair[1]

        # update merges
        merges.append(selected_pair)

        # Update vocab
        idx = len(vocab)
        vocab[idx] = new_byte

        # Update freq_table & pairs_count
        for byte_tuple in list(freq_table.keys()) :
            value = freq_table[byte_tuple]
            if selected_pair in zip(byte_tuple, byte_tuple[1:]):

                # only tuple contains selected_pair need to change
                new_tuple = merge_bytes(byte_tuple, selected_pair)
                freq_table[new_tuple] = freq_table.get(new_tuple, 0) + value
                del freq_table[byte_tuple]

        # Update pairs_count


            
    

    return vocab, merges



input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
vocab_size = 333
special_tokens = ["<|endoftext|>"]

vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

# print(vocab)

# print(merges)

                












