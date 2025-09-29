from cs336_basics.utils.pretokenization import find_boundaries
from cs336_basics.tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm




def preprocessing(data_path, tokenizer, special_tokens, save_path) :
    text_end_token = special_tokens[0]
    end_token = special_tokens[0].encode()
    end_token_len = len(text_end_token)
    full_ids = []
    with open(data_path, "rb") as file :
        boundary_index = find_boundaries(file, end_token, expected_chunks= 100000)
        boundary_index.insert(0, 0) # add starting point as the begin of file

        for (start, end) in tqdm(zip(boundary_index[:-1], boundary_index[1:])) :
            file.seek(start)
            raw_text = file.read(end - start)
            text = raw_text.decode(errors= "ignore")
            text = text[end_token_len:] + text_end_token
            ids = tokenizer.encode(text)
            full_ids += ids

    full_ids = np.array(full_ids, dtype= np.int32)
    full_ids.tofile(save_path)
    print("SAVED ENCODED DATA, a numpy array contains ids !!!")

# data_path = "./data/TinyStoriesV2-GPT4-valid.txt"
# vocab_file = "./data/TinyStoriesV2-GPT4-train_vocab.json"
# merges_file = "./data/TinyStoriesV2-GPT4-train_merges.json"
# special_tokens = ["<|endoftext|>"]
# tokenizer = Tokenizer.from_files(vocab_filepath= vocab_file, merges_filepath= merges_file, special_tokens= special_tokens)


# preprocessing(data_path, tokenizer, special_tokens)