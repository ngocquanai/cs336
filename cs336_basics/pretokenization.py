import os
from typing import BinaryIO
import regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN
import tqdm

def _pretoken(texts: str) :
    pre_tokenized = Counter()
    for text in texts :
        current = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
        pre_tokenized += Counter(current)
    return pre_tokenized


def _find_boundaries(
        file: BinaryIO,
        special_token: bytes,
        expected_chunks: int
) -> list[int] :
    assert isinstance(special_token, bytes), (
        "Special_token must represented as a bytestring"
    )

    file.seek(0, os.SEEK_END)
    file_len = file.tell()
    expected_chunk_size = file_len // expected_chunks
    expected_boundaries = [i*expected_chunk_size for i in range(expected_chunks + 1)]
    expected_boundaries[0] = 0
    expected_boundaries[-1] = file_len
    
    mini_chunk_size = 4096

    for idx in range(1, len(expected_boundaries)) :
        initial_position = expected_boundaries[idx]

        while True :
            file.seek(initial_position)
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"" :
                # it is end of file
                expected_boundaries[idx] = file_len
                break

            found_at = mini_chunk.find(special_token)
            if found_at != -1 :
                expected_boundaries[idx] = initial_position + found_at
                break
            
            # When not found special token in mini chunk
            initial_position += mini_chunk_size

    return sorted(set(expected_boundaries))


def calculate_frequency_table(
        path: str | os.PathLike,
        special_token: bytes,
        PAT: str,
        expected_chunks,
        max_workers: int = 20
) -> dict[tuple[bytes], int]:
    
    text_chunks = []

    with open(path, "rb") as f :
        expected_boundaries = _find_boundaries(f, special_token, expected_chunks)
        occurance = dict()
        special_token_str = special_token.decode()
        
        pbar = tqdm.tqdm(total=len(expected_boundaries), dynamic_ncols=True, mininterval=0)

        for start, end in zip(expected_boundaries[:-1], expected_boundaries[1:]) :

            f.seek(start)
            raw_text = f.read(end - start)
            text = raw_text.decode(errors= "ignore")


            # split by the special token
            texts = text.split(special_token_str)


            text_chunks.append(texts)
            pbar.update(1) 
            pbar.refresh()
            
            

        with ProcessPoolExecutor(max_workers= max_workers) as executor :
            pre_tokenize = executor.map(_pretoken, text_chunks)
        
        occurance = sum(pre_tokenize, Counter())



                    
        frequency_table = dict()
        for key, value in occurance.items() :
            encoded_key = key.encode()
            key_tuple = tuple(bytes([byte_value]) for byte_value in encoded_key)
            frequency_table[key_tuple] = value

        pbar.close()
        return frequency_table








# path = "data/TinyStoriesV2-GPT4-train.txt"
# special_token = "<|endoftext|>".encode("utf-8")
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# frequency_table = calculate_frequency_table(path, special_token, PAT)
# print(len(frequency_table))
            



        

        
        
        




            
            
        



