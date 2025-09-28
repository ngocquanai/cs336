from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("cs336_basics/data/owt_tokenizer")

text = "Hello, I'm NgocQuan from Việt Nam! I love training large language model. <|endoftext|>"

ids = tokenizer.encode(text)
print(type(ids))
print(ids[0])
print(ids)
# for idx in ids :
#     print(tokenizer.decode([idx]))

