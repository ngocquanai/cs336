import regex as re

special_tokens = ["<|endoftext|>", "NGOCQUAN|"]
pattern = "|".join(re.escape(token) for token in special_tokens)

chunk = "hello <|endoftext|> world <|endoftext|> test te||NG|NGNGOCQUAN|NGOCQUAN"
small_chunks = re.split(pattern, chunk)
print("pattern:", pattern)
print("small_chunks:", small_chunks)
