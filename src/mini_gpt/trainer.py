import torch

from mini_gpt.dataset import get_batch


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, eval_iters, block_size, batch_size, device):
    model.eval()

    out = {}

    for split, data in [("train", train_ids), ("val", val_ids)]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(
                data=data,
                block_size=block_size,
                batch_size=batch_size,
                device=device
            )
            _, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out
    
