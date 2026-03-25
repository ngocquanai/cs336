from cs336_basics.utils.constant import GPT2_PRETOKENIZATION
from cs336_basics.utils.io import load_vocab_merges

import regex as re



class Tokenizer() :
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) :

        self.merges = {merge: i for i, merge in enumerate(merges)}


        if special_tokens :
            for token in special_tokens :
                token = token.encode('utf-8')
                if token not in vocab.values() :
                    vocab[len(vocab)] = token
                

        self.vocab = dict()

        self.vocab["int_to_bytes"] = {i : byte_i for i, byte_i in vocab.items()}
        self.vocab["bytes_to_int"] = {byte_i : i for i, byte_i in vocab.items()}

        if special_tokens :
            special_tokens = sorted(special_tokens, key=len, reverse=True)
        
        self.special_tokens = special_tokens

    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) :
        vocab, merges = load_vocab_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        token_ids = []
        if self.special_tokens :
            pattern = '|'.join(re.escape(token) for token in self.special_tokens)
            pattern = f"({pattern})"
            texts = re.split(pattern, text) 
            texts = [chunk for chunk in texts if chunk]
        else :
            texts = [text]
        
        for chunk in texts :
            token_ids += self._encode_chunk(chunk)

        return token_ids

    def encode_iterable(self, iterable) :
        for text in iterable :
            token_ids = self.encode(text)
            for token_id in token_ids :
                yield token_id

    def decode(self, token_ids) -> str:
        decoded_bytes = b""
        for token_id in token_ids :
            decoded_bytes += self.vocab["int_to_bytes"][token_id]
        
        return decoded_bytes.decode('utf-8', errors='replace')

    
    def _encode_chunk(self, text: str) -> list[int]:
        # edge case: special token
        if self.special_tokens and text in self.special_tokens :
            token = text.encode('utf-8')
            token_id = self.vocab["bytes_to_int"][token]
            return [token_id,]


        token_ids = []
        words = re.findall(GPT2_PRETOKENIZATION, text)
        for word in words :
            bytes_list = [bytes([b]) for b in word.encode('utf-8')]

            # Merges until cannot merge
            while len(bytes_list) > 1 :
                pairs_list = [(byte1, byte2) for byte1, byte2 in zip(bytes_list[:-1], bytes_list[1:])]
                merge_pair = min(pairs_list, key= lambda x: self.merges.get(x, float("inf")))
                if merge_pair not in self.merges :
                    break
                bytes_list = self.merge(bytes_list, merge_pair)

            # Convert bytes to int
            for bytes_token in bytes_list :
                token_id = self.vocab["bytes_to_int"][bytes_token]
                token_ids.append(token_id)

        return token_ids




    def merge(self, bytes_list: list, merge_pair: tuple) :
        idx = 0
        new_bytes_list = []
        while idx < len(bytes_list) :
            if idx + 1 < len(bytes_list) and (bytes_list[idx], bytes_list[idx+1]) == merge_pair :
                new_bytes_list.append(merge_pair[0] + merge_pair[1])
                idx += 2
            else :
                new_bytes_list.append(bytes_list[idx])
                idx += 1

        return new_bytes_list
    

        

# special_tokens = ["<|endoftext|>", "<|end|>"]
# tokenizer = Tokenizer.from_files("../data/TinyStories_vocab.json", "../data/TinyStories_merges.txt", special_tokens= special_tokens)

# text = "Then, in early 2009, <|endoftext|> the Moseleys <|end|> heard that the downtown Holiday"
# text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

# ids = tokenizer.encode(text)
# print(ids)
# text = tokenizer.decode(ids)
# print(text)
