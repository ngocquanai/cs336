import torch
from torch.utils.data import Dataset
import os




from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils.function import softmax, cross_entropy
from cs336_basics.utils.io import save_vocab_and_merge
from cs336_basics.train_bpe import train_bpe
from cs336_basics.model import TransformerLM
from cs336_basics.utils.data import preprocessing

from cs336_basics.dataset import DatasetLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR




def decoding(model, tokenizer, prompt, temperature= 1, max_token_generated = 256, context_len= 256, end_tokens= ["<|endoftext|>"]) :
    end_token_ids = tokenizer.encode(end_tokens)

    generated_tokens = 0

    input_ids = tokenizer.encode(prompt)
    initial_len = len(input_ids)

    pred = model(input_ids)
    generated_tokens += 1


    while pred not in end_token_ids :
        token_id = int(torch.argmax(softmax(pred, temperature= temperature)))
        input_ids.append(token_id)
        if len(input_ids) >= context_len :
            pred = model(input_ids[-context_len:])
        else :
            pred = model(input_ids)

        generated_tokens += 1

        if generated_tokens >= max_token_generated :
            break

    generated_ids = input_ids[initial_len:]
    output = tokenizer.decode(generated_ids)

    return output

def eval(model, optimizer, scheduler, dataset, batch_size=32) :
    dataset_len = dataset.get_len(split= "valid")
    steps = dataset_len // batch_size
    total_loss = 0
    for _ in range(steps) :
        inputs, labels = dataset.sample("valid", batch_size= batch_size)
        logits = model(inputs)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        total_loss += loss
    
    final_loss = total_loss / steps
    return final_loss




device = torch.device("cpu")
vocab_size = 10000    
context_length = 256
num_layers = 4
d_model = 512
d_ff = 1344
num_heads = 16
rope_theta = 10000.0
steps = 50000
batch_size= 32
lr = 1e-3
wd = 0.01
betas = (0.9, 0.99)
T_max = 3

special_tokens = ["<|endoftext|>"]
train_data_path = "data/TinyStoriesV2-GPT4-train.txt"
valid_data_path = "data/TinyStoriesV2-GPT4-valid.txt"

vocab_path = "./data/vocab.json"
merges_path = "./data/merges.json"

encoded_train_path = "./data/encode_train.bin"
encoded_valid_path = "./data/encode_valid.bin"

if not os.path.exists(vocab_path) :
    vocab, merges = train_bpe(input_path= train_data_path, vocab_size= vocab_size, special_tokens= special_tokens)
    save_vocab_and_merge(vocab= vocab, merges= merges, vocab_path= vocab_path, merges_path= merges_path)


tokenizer = Tokenizer.from_files(vocab_filepath= vocab_path, merges_filepath= merges_path, special_tokens= special_tokens)

if not os.path.exists(encoded_train_path) :
    preprocessing(train_data_path, tokenizer= tokenizer, special_tokens= special_tokens, save_path= encoded_train_path)
    preprocessing(valid_data_path, tokenizer= tokenizer, special_tokens= special_tokens, save_path= encoded_valid_path)


dataset = DatasetLM(train_path= encoded_train_path, valid_path= encoded_valid_path, batch_size= batch_size, context_len= context_length, device= device)




model = TransformerLM(vocab_size= vocab_size, context_length= context_length, num_layers= num_layers,
                      d_model= d_model, num_heads= num_heads, d_ff= d_ff, rope_theta= rope_theta, device= device)

optimizer = AdamW(model.parameters(), lr= lr, betas= betas, weight_decay= wd)
scheduler = CosineAnnealingLR(optimizer= optimizer, T_max= T_max)

# Start training

total_loss = 0
for step in range(steps) :
    inputs, labels = dataset.sample("train")
    logits = model(inputs)
    loss = cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    total_loss += loss
    if step % 100 == 0 and step < 10000:
        print(f"Loss at step {step}: {loss}")

    if step % 250 == 0 :
        eval_loss = eval(model, optimizer, scheduler, dataset, batch_size= 2* batch_size)
        eval_loss = int(eval_loss * 10000)/10000
        print(f"Evaluation!!! Eval loss at step {step} is: {eval_loss}")
















    

    


    



