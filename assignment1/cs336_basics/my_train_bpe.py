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

def _merge_bytes(byte_tuple, merge_loc):
    """
    Merge the byte tuple at the merge location.
    """
    assert len(byte_tuple) > 1, "Cannot merge a byte tuple with length less than 2."
    prefix = byte_tuple[:merge_loc]
    tomerge = byte_tuple[merge_loc:merge_loc+2]
    suffix = byte_tuple[merge_loc+2:]
    new_byte_tuple = prefix + (b"".join(tomerge),) + suffix
    return new_byte_tuple, prefix, suffix


def adjacent_pair(freq_table: dict) :

    pairs_count = dict()

    for word, count in freq_table.items() :
        for byte1, byte2 in zip(word[:-1], word[1:]) :
            pair = (byte1, byte2)
            pairs_count[pair] = pairs_count.get(pair, 0) + count
    return pairs_count

@profile
def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_processes= 1) :

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
        if not pairs_count :
            break
        selected_pair = most_frequent_pair(pairs_count)
        new_byte = selected_pair[0] + selected_pair[1]

        # update merges
        merges.append(selected_pair)

        # Update vocab
        new_vocab_idx = len(vocab)
        vocab[new_vocab_idx] = new_byte

        # Update the pre-token frequency table and pairs_count
        new_freq_table = {}
        for byte_tuple, count in freq_table.items():
            i=0
            while i < len(byte_tuple):
                pair = byte_tuple[i:i+2]
                if pair == selected_pair:
                    byte_tuple, prefix, suffix = _merge_bytes(byte_tuple, i)

                    # Update the pair frequency table
                    if prefix:
                        add_pair = (prefix[-1], vocab[new_vocab_idx])
                        pairs_count[add_pair] = pairs_count.get(add_pair, 0) + count
                        del_pair = (prefix[-1], selected_pair[0])
                        pairs_count[del_pair] -= count
                    if suffix:
                        add_pair = (vocab[new_vocab_idx], suffix[0])
                        pairs_count[add_pair] = pairs_count.get(add_pair, 0) + count
                        del_pair = (selected_pair[1], suffix[0])
                        pairs_count[del_pair] -= count
                    pairs_count[selected_pair] -= count
                i+=1
            # Update the pre-token frequency table
            new_freq_table[byte_tuple] = count
        freq_table = new_freq_table
                


            
    
    # print(len(vocab))
    # print("*"*300)
    return vocab, merges



input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
vocab_size = 555
special_tokens = ["<|endoftext|>"]

vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

# print(vocab)

# print(merges)

                












