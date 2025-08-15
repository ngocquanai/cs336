from cs336_basics.utils.io import get_tokenizer_from_vocab_merges_path, GPT2_PRETOKENIZER_PATTERN
import regex as re
# from typing import Iterable, Dict, Tuple, List

def get_pairs(ids: list[int]) -> set:
    pairs = set()
    for pair in zip(ids[:-1], ids[1:]) :
        pairs.add(pair)


    return pairs

def update_ids(ids: list[int], pair: tuple[int, int], new_id: int) :
    idx = 0
    while  idx < (len(ids)-1) :
        if (ids[idx], ids[idx+1]) == pair :
            ids = ids[:idx] + [new_id] + ids[idx+2:]

        idx += 1
    
    return ids




class Tokenizer() :
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None): 
        int_to_byte = vocab
        byte_to_int = {value: key for key, value in vocab.items()}

        # set up vocab
        self.vocab = {}
        self.vocab["int_to_byte"] = int_to_byte
        self.vocab["byte_to_int"] = byte_to_int
        self._refine_vocab()

        # set up merges
        self.merges = dict()
        for b1, b2 in merges :
            int1, int2 = self.vocab["byte_to_int"][b1], self.vocab["byte_to_int"][b2]
            self.merges[(int1, int2)] = self.vocab["byte_to_int"][b1+b2]





        # add special tokens as string to id mapping
        self.special_tokens = {}
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_byte = token.encode("utf-8")
                if token_byte not in self.vocab['byte_to_int']:
                    self.vocab['byte_to_int'][token_byte] = len(self.vocab['byte_to_int'])
                    self.vocab['int_to_byte'][len(self.vocab['int_to_byte'])] = token_byte
                    self.special_tokens[token] = len(self.vocab['int_to_byte'])
                else:
                    self.special_tokens[token] = self.vocab['byte_to_int'][token_byte]


    @classmethod
    def from_files(cls, vocab_filepath: str,
                    merges_filepath: str, special_tokens= None):
        """Load vocab and merges from file paths"""

        vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_filepath, merges_filepath)

        return cls(vocab, merges, special_tokens)
    

    def _refine_vocab(self) :
        """Ensure the vocab contain all bytes"""
        for i in range(256) :
            byte = bytes([i])
            if byte not in self.vocab['byte_to_int'] :
                idx = len(self.vocab['byte_to_int'])
                self.vocab['byte_to_int'][byte] = idx
                self.vocab['int_to_byte'][idx] = byte
        
    
    def _encode_chunk(self, text: str) -> bytes:
        """Encode the text chunk, no special_tokens include"""

        if text in self.special_tokens:
            return [self.special_tokens[text]] # type: ignore
        

        # ELSE

        text_chunks = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
        output_ids = []
        for chunk in text_chunks :
            ids = [self.vocab["byte_to_int"][bytes([b])] for b in chunk.encode()]

            while len(ids) >= 2 :
                pairs = get_pairs(ids)

                priority_pair = min(pairs, key= lambda pair: self.merges.get(pair, float("inf")))
                if priority_pair not in self.merges :
                    break

                # merges this priority pair
                new_id = self.merges[priority_pair]
                ids = update_ids(ids, priority_pair, new_id)

            
            output_ids.extend(ids)

        return output_ids # type: ignore
    
    
    def encode(self, text: str) -> list:
        """Encode the input text, return bytes"""

        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            special_split_chunk = re.split(special_pattern, text)
        else:
            special_split_chunk = [text]
        
        ids = []
        for chunk in special_split_chunk :
            ids += self._encode_chunk(chunk)
        
        return ids
    

    def encode_iterable(self, texts: list[str]) :
        for text in texts :
            ids = self.encode(text)
            for id in ids :
                yield id




    def decode(self, ids: list[int]) -> str:
        """Decode the bytes, return text"""
        total_bytes = b""
        for id in ids :
            byte = self.vocab["int_to_byte"][id]
            total_bytes += byte
        
        text = total_bytes.decode(errors='replace')
        return text            

    
    

tokenizer = Tokenizer.from_files("./data/TinyStoriesV2-GPT4-train_vocab.json", "./data/TinyStoriesV2-GPT4-train_merges.json")


text = "Hey, I'm NgocQuan, from Vietnam."   


byte = (tokenizer.encode(text))

print(tokenizer.decode(byte))
        
         