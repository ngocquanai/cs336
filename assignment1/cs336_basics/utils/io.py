import json
import base64
import torch
import os


### save() and load() function are not my own implementation
def save_vocab_merges(vocab, merges, vocab_path, merges_path) :
    """
    Persist vocab and merges to disk.

    vocab_path  : JSON file  { "<int idx>" : "<base64-encoded bytes>" }
    merges_path : text file, one merge per line: "<hex_a> <hex_b>"
                  where hex_a / hex_b are the two byte-sequences being merged,
                  encoded as hex strings (e.g. "68656c6c6f 20776f726c64").
    """
    # --- vocab ---
    serialisable = {
        str(idx): base64.b64encode(token_bytes).decode('ascii')
        for idx, token_bytes in vocab.items()
    }
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(serialisable, f, indent=2)

    # --- merges ---
    with open(merges_path, 'w', encoding='utf-8') as f:
        for left, right in merges:
            f.write(f"{left.hex()} {right.hex()}\n")


def load_vocab_merges(vocab_path, merges_path):
    """
    Inverse of save().
    Returns vocab  : dict[int, bytes]
            merges : list[tuple[bytes, bytes]]
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    vocab = {int(idx): base64.b64decode(b64) for idx, b64 in raw.items()}

    merges = []
    with open(merges_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left_hex, right_hex = line.split()
            merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))

    return vocab, merges


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
    iteration: int, out: str|os.PathLike
    ) :
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
        }

    torch.save(checkpoint, out)

def load_checkpoint(
    src: str|os.PathLike, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) :
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
