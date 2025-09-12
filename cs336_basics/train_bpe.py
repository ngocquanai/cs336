from cs336_basics.utils.pretokenization import calculate_frequency_table

from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN, save_vocab_and_merge

from tqdm import tqdm
import os
import time
print("START")
start = time.time()
def _update_tuple(bytes_tuple, merge_id) :
    # merge 2 bytes at index merge_int, merge_int+1 into one.

    need_to_merge = bytes_tuple[merge_id:merge_id+2]
    merged = b"".join(need_to_merge)
    prefix = bytes_tuple[:merge_id]
    suffix = bytes_tuple[merge_id+2:]

    new_tuple = prefix + (merged,) + suffix

    return new_tuple, prefix, suffix 
   


def train_bpe(input_path: str | os.PathLike,
              vocab_size: int, 
              special_tokens: list[str],
              progress_bar: bool = False, 
              expected_chunks= 50000
              ) : 
    special_token = special_tokens[0].encode()
    frequency_table = calculate_frequency_table(input_path, special_token, GPT2_PRETOKENIZER_PATTERN, expected_chunks= expected_chunks) # dict[tuple[bytes], int]
    print(len(frequency_table))
    adjacent_count = dict()
    merges = []



    # init vocabulary
    vocabulary = dict() # dict[int, bytes]
    for i in range(256) :
        vocabulary[i] = bytes([i])
    vocabulary[256] = special_token

    initial_len = len(vocabulary)
    pbar = tqdm(total= initial_len) 
    


# first, compute original adjacent_count
    for bytes_tuple, times in frequency_table.items() :
        for a, b in zip(bytes_tuple, bytes_tuple[1:]) :
            adj = (a, b)
            if adj not in adjacent_count :
                adjacent_count[adj] = times 
            else :
                adjacent_count[adj] += times
        

    while len(vocabulary) < vocab_size :

        new_id = max(vocabulary.keys()) + 1
        most_freq_adj = max(adjacent_count, key=lambda k: (adjacent_count[k], k))
        new_token = b"".join(most_freq_adj)


        # add new token
        vocabulary[new_id] = new_token
        merges.append(most_freq_adj)


        new_frequency_table = {}

        for bytes_tuple, times in frequency_table.items() :
            i = 0
            while i < len(bytes_tuple) - 1 :
                pair = bytes_tuple[i:i+2]
                if pair == most_freq_adj :
                    # update tuple
                    bytes_tuple, prefix, suffix = _update_tuple(bytes_tuple, i)


                    # update adjacent count
                    if prefix:
                        add_pair = (prefix[-1], vocabulary[new_id])
                        adjacent_count[add_pair] = adjacent_count.get(add_pair, 0) + times
                        del_pair = (prefix[-1], most_freq_adj[0])
                        adjacent_count[del_pair] -= times
                    if suffix:
                        add_pair = (vocabulary[new_id], suffix[0])
                        adjacent_count[add_pair] = adjacent_count.get(add_pair, 0) + times
                        del_pair = (most_freq_adj[1], suffix[0])
                        adjacent_count[del_pair] -= times
                    adjacent_count[most_freq_adj] -= times

                # Not a most_freq_adj pair
                i += 1
            
            new_frequency_table[bytes_tuple] = times
        frequency_table = new_frequency_table

        pbar.update(len(vocabulary) - initial_len - pbar.n) 



    pbar.close() 
    print(len(vocabulary))
    return vocabulary, merges


# path = "data/TinyStoriesV2-GPT4-train.txt"
# special_token = "<|endoftext|>"
# special_tokens = [special_token,]

# vocab, merges = train_bpe(path, 15000, special_tokens, expected_chunks= 15000)

# end = time.time()

# print("Processing time: ", int((end-start)*1000)/1000/60, "mins. Thanks!")

# print("Saving...")
# save_vocab_and_merge(vocab, merges, "./data/TinyStoriesV2-GPT4-train_vocab.json", "./data/TinyStoriesV2-GPT4-train_merges.json")
# print("Saved!")


            
                














        
