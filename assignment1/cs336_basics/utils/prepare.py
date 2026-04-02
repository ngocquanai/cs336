# First, create an Byte-Pair Encoding, save as vocab.json and merges.txt
from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils.io import save_vocab_merges
from cs336_basics.utils.data import text_to_ids


def create_bpe(text_file_path: str, vocab_size: int, special_tokens: list, saved_vocab_path: str, saved_merges_path: str, num_processes=128) :

    vocab, merges = train_bpe(text_file_path, vocab_size, special_tokens, num_processes= num_processes)

    save_vocab_merges(vocab, merges, vocab_path= saved_vocab_path, merges_path= saved_merges_path)


# First, create an Byte-Pair Encoding, save as vocab.json and merges.txt

text_file_path = "../data/TinyStoriesV2-GPT4-train.txt"

vocab_path = "../data/TinyStories_vocab.json"
merges_path = "../data/TinyStories_merges.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]

# create_bpe(text_file_path, vocab_size, special_tokens, saved_vocab_path= vocab_path, saved_merges_path= merges_path, num_processes= 128)



# Then, tokenize entire dataset and save to a npy file.
train_dataset = "../data/owt_train.txt"
val_dataset = "../data/owt_valid.txt"
vocab_path = "../data/owt_vocab.json"
merges_path = "../data/owt_merges.txt"
special_tokens = ["<|endoftext|>"]
saved_train_token_ids = "../data/owt-train_token_ids.npy"
saved_val_token_ids = "../data/owt-val_token_ids.npy"

print("Start")
text_to_ids(text_file= train_dataset, vocab_file= vocab_path, merges_file= merges_path, saved_path= saved_train_token_ids, special_tokens= special_tokens, num_processes= 128)
print("Finish tokenizing train dataset")

text_to_ids(text_file= val_dataset, vocab_file= vocab_path, merges_file= merges_path, saved_path= saved_val_token_ids, special_tokens= special_tokens, num_processes= 128)

print(saved_train_token_ids, saved_val_token_ids)


