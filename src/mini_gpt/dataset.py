import torch


def build_train_val_ids(tokenizer, text, train_ratio=0.9):
    ids = tokenizer.encode(text)
    ids = torch.tensor(ids, dtype=torch.long)

    split_idx = int(len(ids) * train_ratio)
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    return train_ids, val_ids


def get_batch(data, block_size, batch_size, device):
    """
    data: shape [N]
    return:
        x: [B, T]
        y: [B, T]
    """
    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start + 1, (batch_size,))

    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])

    x = x.to(device)
    y = y.to(device)

    return x, y
