import argparse
from pickletools import optimize
import torch
import os

from torch.utils.data import dataloader
from cs336_basics.dataloader import DataLoader
from cs336_basics.layers import TransformerLM

from cs336_basics.optimizer import AdamW, SGD, WarmupCosineAnnealingLR
from cs336_basics.loss import CrossEntropyLoss
from cs336_basics.utils.io import load_checkpoint, save_checkpoint


def parse_args() :
    parser = argparse.ArgumentParser(description= "Training Transformer Language Model on given dataset")
    # TYPE, REQUIRED, DEFAULT, ACTION= "store_true", CHOICES

    # data
    parser.add_argument("--train_path", type= str, required= True)
    parser.add_argument("--valid_path", type= str, required= True)

    # model
    parser.add_argument("--vocab_size", type= int, required= True)
    parser.add_argument("--context_len", type= int, required= True, help= "Context length of the model")
    parser.add_argument("--num_layers", type= int, required= True)
    parser.add_argument("--d_model", type= int, required= True)
    parser.add_argument("--d_ff", type= int, required= True)
    parser.add_argument("--num_heads", type= int, required= True)
    parser.add_argument("--rope_theta", type= float, default= 10000.0, help= "Theta value in Rotary Positional Embedding")
    parser.add_argument("--dtype", type= str, default= "float32", choices= ["float32", "float64", "float16", "bfloat16"])


    # training
    parser.add_argument("--batch_size", type= int, default= 32)
    parser.add_argument("--device", type= str, default= "cpu", choices= ["cpu", "cuda"])
    parser.add_argument("--lr", type= float, required= True, help= "learning rate")
    parser.add_argument("--lr_scheduler", type= str, default= "cosine", choices= ["cosine"])
    parser.add_argument("--max_steps", type= int, required= True)


    # optimizer
    parser.add_argument("--optimizer", type= str, default= "adamw", choices= ["adamw", "sgd", "sam", "muon"])
    parser.add_argument("--betas", type= float, nargs= 2, default= [0.9, 0.99], help= "beta value for moment estimation in optimizer")
    parser.add_argument("--weight_decay", type= float, default= 0.1)
    parser.add_argument("--optim_eps", type=float, default= 1e-8)

    # learning rate scheduler
    parser.add_argument("--min_lr", type= float, default= 1e-5)
    parser.add_argument("--max_lr", type= float, default= 1e-3)
    parser.add_argument("--T_warmup", type= int, default= 10000)
    parser.add_argument("--T_cosine", type= int, default= 100000)


    # eval
    parser.add_argument("--eval_period", type= int, default= 1000)
    parser.add_argument("--eval_batch", type= int, default= 100, help="In each validation, we random selects --eval_batch batch from eval_dataloader")

    # save checkpoint
    parser.add_argument("--saved_path", type= str, default= "./checkpoints")
    parser.add_argument("--save_period", type= int, default= 1000, help= "Save intermediate checkpoints after how many steps")

    return parser.parse_args()


def train(args) :

    os.makedirs(args.saved_path, exist_ok= True)
    device = torch.device(args.device)
    if args.dtype == "float64" :
        dtype = torch.float64
    elif args.dtype == "float32" :
        dtype = torch.float32
    elif args.dtype == "float16" :
        dtype = torch.float16
    elif args.dtype == "bfloat16" :
        dtype = torch.bfloat16
    else :
        raise ValueError(f"Invalid torch dtype {args.dtype}")
    

    # data
    train_dataloader = DataLoader(
        datapath= args.train_path,
        batch_size= args.batch_size,
        context_len= args.context_len,
        device= device)

    valid_dataloader = DataLoader(
        datapath= args.valid_path,
        batch_size= args.batch_size,
        context_len= args.context_len,
        device= device)

    # model

    model = TransformerLM(
        vocab_size= args.vocab_size,
        context_length= args.context_len,
        num_layers= args.num_layers,
        d_model= args.d_model,
        num_heads= args.num_heads,
        d_ff= args.d_ff,
        rope_theta= args.rope_theta,
        device= device,
        dtype= dtype
    )

    # optimizer & scheduler
    if args.optimizer == "adamw" :
        betas = tuple(args.betas)
        optimizer = AdamW(params= model.parameters(), lr= args.lr, betas= betas, weight_decay= args.weight_decay, eps = args.optim_eps)
        scheduler = WarmupCosineAnnealingLR(optimizer= optimizer, alpha_min= args.min_lr, alpha_max= args.max_lr, T_warmup= args.T_warmup, T_cosine= args.T_cosine)

    elif args.optimizer == "sgd" :
        optimizer = SGD(params=model.parameters(), lr = args.lr)
        scheduler = WarmupCosineAnnealingLR(optimizer= optimizer, alpha_min= args.min_lr, alpha_max= args.max_lr, T_warmup= args.T_warmup, T_cosine= args.T_cosine)
    else :
        raise ValueError(f"Do not support Optimizer {args.optimizer}!")

    
    # TRAINING
    model.train()
    criteria = CrossEntropyLoss()
    for step in range(args.max_steps) :
        x, y = next(train_dataloader)
        logits = model(x)
        loss = criteria(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step + 1 % 10 == 0 :
            print(f"Loss at step {step}: {int(loss.item()*10000)/10000}")


        # validation
        if step + 1 % args.eval_period == 0 :
            model.eval()
            val_loss = eval(model, dataloader, criteria, args.eval_batch)
            model.train()

        # save checkpoint
        if step + 1 % args.save_period == 0 :
            saved_path = os.path.join(args.saved_path, f"ckpt_step_{step+1}.pt")
            save_checkpoint(model, optimizer, iteration= step + 1, out= saved_path)

    # End of training



def eval(model, dataloader, criteria, steps) :
    total_loss = 0
    for step in range(steps) :
        x, y = next(dataloader)
        logits = model(x)
        loss = criteria(logits, y)
        total_loss += loss
    
    return total_loss/steps


if __name__ == "__main__" :
    args = parse_args()
    train(args)









    



    
    
