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


def find_boundaries(
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


# Pass this dictionary to the worker function, as the worker processes don't 
# have access to the variables in the main function's scope.
WORKER_GLOBALS = {}

def _process_chunk(start_end_pair):
    """Worker function that processes a single file chunk by reading it directly."""
    # Note: WORKER_GLOBALS must be set up before ProcessPoolExecutor starts
    path = WORKER_GLOBALS['path']
    special_token_str = WORKER_GLOBALS['special_token_str']
    
    start, end = start_end_pair
    
    try:
        with open(path, "rb") as f:
            f.seek(start)
            raw_text = f.read(end - start)
            text = raw_text.decode(errors="ignore")
            
            # Split and pre-tokenize this single chunk
            texts = text.split(special_token_str)
            
            pre_tokenized = Counter()
            for t in texts:
                current = re.findall(GPT2_PRETOKENIZER_PATTERN, t)
                pre_tokenized += Counter(current)
            
            return pre_tokenized
    except Exception as e:
        # Crucial for debugging OOM or other hidden errors
        print(f"Worker ERROR processing chunk {start}-{end}: {e}", flush=True)
        raise



# def calculate_frequency_table(
#         path: str | os.PathLike,
#         special_token: bytes,
#         PAT: str,
#         expected_chunks,
#         max_workers: int = 2
# ) -> dict[tuple[bytes], int]:
    
#     text_chunks = []

#     with open(path, "rb") as f :
#         expected_boundaries = find_boundaries(f, special_token, expected_chunks)
#         occurance = dict()
#         special_token_str = special_token.decode()
        
#         pbar = tqdm.tqdm(total=len(expected_boundaries), dynamic_ncols=True, mininterval=0)
#         print("Point 0")
#         print("Max workers: ", max_workers)
#         for start, end in zip(expected_boundaries[:-1], expected_boundaries[1:]) :

#             f.seek(start)
#             raw_text = f.read(end - start)
#             text = raw_text.decode(errors= "ignore")


#             # split by the special token
#             texts = text.split(special_token_str)


#             text_chunks.append(texts)
#             pbar.update(1) 
#             pbar.refresh()
            
            
#         print("Point 1")
#         with ProcessPoolExecutor(max_workers= max_workers) as executor :
#             pre_tokenize = executor.map(_pretoken, text_chunks)
#         print("Point 2")
#         occurance = sum(pre_tokenize, Counter())



#         print("Point 3")
#         frequency_table = dict()
#         for key, value in occurance.items() :
#             encoded_key = key.encode()
#             key_tuple = tuple(bytes([byte_value]) for byte_value in encoded_key)
#             frequency_table[key_tuple] = value
#         print("Point 4")
#         pbar.close()
#         return frequency_table

def calculate_frequency_table(
        path: str | os.PathLike,
        special_token: bytes,
        PAT: str,
        expected_chunks,
        max_workers: int = 2
) -> dict[tuple[bytes], int]:
    
    # --- STEP 1: Find Boundaries ---
    with open(path, "rb") as f :
        expected_boundaries = find_boundaries(f, special_token, expected_chunks)
    
    pbar = tqdm.tqdm(total=len(expected_boundaries) - 1, dynamic_ncols=True, mininterval=0)
    print("Point 0: Found boundaries.")
    print("Max workers: ", max_workers)

    # --- STEP 2: Prepare Worker Input (Lightweight) ---
    # The input to map is a list of (start, end) tuples, which is tiny.
    boundaries_pairs = list(zip(expected_boundaries[:-1], expected_boundaries[1:]))

    # --- STEP 3: Setup Worker Global Variables ---
    # Workers need access to the file path and special token string.
    global WORKER_GLOBALS
    WORKER_GLOBALS['path'] = path
    WORKER_GLOBALS['special_token_str'] = special_token.decode()
    
    print("Point 1: Starting parallel execution.")
    
    # --- STEP 4: Parallel Processing (File I/O is now inside the worker) ---
    with ProcessPoolExecutor(max_workers=max_workers) as executor :
        # Use map to process the lightweight boundary pairs
        pre_tokenize_results = list(executor.map(_process_chunk, boundaries_pairs))
        
        # Use a secondary progress bar for the actual processing
        # Note: map is blocking, so the overall tqdm will be updated *after* map finishes.
        # It's better to use futures.as_completed for progress, but this is simpler.

    print("Point 2: Parallel execution complete.")
    
    # --- STEP 5: Aggregate Results ---
    # Sum the list of Counter objects
    occurance = sum(pre_tokenize_results, Counter())

    # The rest of the logic remains the same...
    print("Point 3: Aggregating results.")
    frequency_table = {}
    for key, value in occurance.items() :
        encoded_key = key.encode()
        # Your original logic converts the encoded key into a tuple of single-byte bytes objects
        key_tuple = tuple(bytes([byte_value]) for byte_value in encoded_key)
        frequency_table[key_tuple] = value
    
    print("Point 4: Frequency table prepared.")
    pbar.close() # Close the tqdm that was showing boundary search progress
    return frequency_table






# path = "data/TinyStoriesV2-GPT4-train.txt"
# special_token = "<|endoftext|>".encode("utf-8")
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# frequency_table = calculate_frequency_table(path, special_token, PAT)
# print(len(frequency_table))
            



        

        
        
        




            
            
        



