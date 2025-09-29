import torch
from torch.utils.data import Dataset
import os
import torch.nn as nn
import torch.nn.functional as F




from cs336_basics.tokenizer import train_tokenizer
from cs336_basics.utils.function import softmax, cross_entropy
from cs336_basics.utils.io import save_vocab_and_merge
from cs336_basics.train_bpe import train_bpe
from cs336_basics.model import TransformerLM
from cs336_basics.utils.data import preprocessing

from cs336_basics.dataset import DatasetLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import PreTrainedTokenizerFast

from tqdm import tqdm

def get_last_token_id(logits: torch.Tensor, temperature) -> int :
    pred = softmax(logits, temperature= temperature)
    last_token_pred = pred[:, -1, :]
    token_id = torch.argmax(last_token_pred).item()
    return token_id


def decoding(model, tokenizer, prompt, temperature= 1, max_token_generated = 256, context_len= 256, end_tokens= ["<|endoftext|>"]) :
    end_token_ids = tokenizer.encode(end_tokens[0])

    generated_tokens = 0

    input_ids = tokenizer.encode(prompt)
    initial_len = len(input_ids)

    logits = model(torch.tensor(input_ids).reshape(1, -1)) # Full token predictions
    token_id = get_last_token_id(logits, temperature)
    generated_tokens += 1

    # breakpoint()
    while token_id not in end_token_ids :

        input_ids.append(token_id)

        if len(input_ids) >= context_len :
            logits = model(torch.tensor(input_ids[-context_len:]).reshape(1, -1))
        else :
            logits = model(torch.tensor(input_ids).reshape(1, -1))

        token_id = get_last_token_id(logits, temperature) # When Inference, We only care about the last token 

        generated_tokens += 1

        if generated_tokens >= max_token_generated :
            break

    generated_ids = input_ids[initial_len:]
    output = tokenizer.decode(generated_ids)

    return output

def eval(model, optimizer, scheduler, dataset, batch_size=32) :

    model.eval()
    dataset_len = dataset.get_len(split= "valid")
    # steps = dataset_len // (batch_size * 2)
    steps = 236
    total_loss = 0

    total_correct_predictions = 0
    total_tokens = 0
        
    with torch.no_grad() :
        for _ in range(steps) :
            inputs, labels = dataset.sample("valid", batch_size= batch_size)
            logits = model(inputs)

            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)

            loss = cross_entropy(logits, labels)
            total_loss += loss.item()

            predicted_tokens = torch.argmax(logits, dim=-1)
            correct_predictions = (predicted_tokens == labels)

            total_correct_predictions += correct_predictions.sum().item()
            total_tokens += labels.numel() # Total elements in the labels tensor


    model.train()



    final_loss = total_loss / steps
    # Calculate accuracy
    accuracy = total_correct_predictions / total_tokens if total_tokens > 0 else 0.0

    return final_loss, accuracy




device = torch.device("cuda")
vocab_size = 32768  
context_length = 256
num_layers = 8
d_model = 768
d_ff = 1344
num_heads = 12
rope_theta = 10000.0
steps = 100000
batch_size= 16
lr = 5e-4
wd = 0.01
betas = (0.9, 0.99)
T_max = 3

special_tokens = ["<|endoftext|>"]
train_data_path = "./cs336_basics/data/owt_train.txt"
valid_data_path = "./cs336_basics/data/owt_valid.txt"

tokenizer_path = "./cs336_basics/data/owt_tokenizer"

encoded_train_path = "./cs336_basics/data/encode_train.bin"
encoded_valid_path = "./cs336_basics/data/encode_valid.bin"


print("Start! Hope it work well!")
# if not os.path.exists(vocab_path) :
#     vocab, merges = train_bpe(input_path= train_data_path, vocab_size= vocab_size, special_tokens= special_tokens)
#     save_vocab_and_merge(vocab= vocab, merges= merges, vocab_path= vocab_path, merges_path= merges_path)

# print("Loaded vocab and merges")

# tokenizer = Tokenizer.from_files(vocab_filepath= vocab_path, merges_filepath= merges_path, special_tokens= special_tokens)



# For open web dataset, using Tokenizer from huggingface instead

if not os.path.exists(tokenizer_path) :
    train_tokenizer(train_data_path, tokenizer_path)
else :
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)



test_str = "Hello, I'm NgocQuan from Việt Nam! I love training large language model. <|endoftext|>"
print(f"Test tokenizer: {test_str}")
encoded_ids = tokenizer.encode(test_str)
print(f"Encoded ids contains {len(encoded_ids)} tokens: {encoded_ids}")


print(f"Decoded text: {tokenizer.decode(encoded_ids)}")

print("Loaded tokenizer!")
if not os.path.exists(encoded_train_path) :
    preprocessing(train_data_path, tokenizer= tokenizer, special_tokens= special_tokens, save_path= encoded_train_path)
    preprocessing(valid_data_path, tokenizer= tokenizer, special_tokens= special_tokens, save_path= encoded_valid_path)


dataset = DatasetLM(train_path= encoded_train_path, valid_path= encoded_valid_path, batch_size= batch_size, context_len= context_length, device= device)
print("Loaded dataset!")



model = TransformerLM(vocab_size= vocab_size, context_length= context_length, num_layers= num_layers,
                      d_model= d_model, num_heads= num_heads, d_ff= d_ff, rope_theta= rope_theta, device= device)
model = model.to("cuda")
optimizer = AdamW(model.parameters(), lr= lr, betas= betas, weight_decay= wd)
scheduler = CosineAnnealingLR(optimizer= optimizer, T_max= T_max)
print("Loaded model!")
# Start training
criterion = nn.CrossEntropyLoss()
total_loss = 0
print("Training time !!!")
torch.autograd.set_detect_anomaly(True)
for step in range(steps) :
    inputs, labels = dataset.sample("train", batch_size= batch_size)

    inputs = inputs.to("cuda")
    labels = labels.to("cuda")
    # samples = inputs[0].tolist()
    # print(tokenizer.decode(samples))
    
    logits = model(inputs)
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)

    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    total_loss += loss.item()
    if step % 250 == 0 :
        loss = int(loss * 10000) / 10000
        print(f"Loss at step {step}: {loss}")

    if step % 500 == 236 :
        prompt = "The capital city of Korea is:"
        test_ans = decoding(model, tokenizer, prompt, temperature= 1, max_token_generated= 100)
        print(prompt)
        answer = f"Answer: {test_ans} <end>"
        print(repr(answer))
        print("-"*50)
        eval_loss, eval_acc = eval(model, optimizer, scheduler, dataset, batch_size= 2* batch_size)
        eval_loss = int(eval_loss * 10000)/10000
        eval_acc = int(eval_acc * 10000) / 10000
        print(f"Eval loss at step {step} is: {eval_loss}; accuracy: {eval_acc*100}")
















    

    


    



