# pytest /home/groups/candes/zitong/cs336-assignment1-basics/tests/test_train_bpe.py
import regex as re
from typing import Iterable
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from collections import Counter
import concurrent.futures

GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

text = "âShortly thereafter, with utility costs mounting and many floors vacant, the Moseleys saw an opportunity."
words = re.findall(GPT2_PRETOKENIZER_PATTERN, text)

ans = (bytes([b]) for b in text.encode('utf-8'))

for b in text.encode('utf-8') :
    print(bytes([b]))
print(ans)