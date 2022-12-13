import os
import pickle
import sys
import torch
import torch.nn as nn

from torch.optim import Adam as AdamW
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup
from transformers import GPTNeoModel, GPTNeoForCausalLM

from data_loader import ClipPoemDataset
from gpt_mlp.gpt_clip_model import GPTMLPModel
from gpt_transformer.gpt_transformer_model import GPTTransformerModel
from tqdm import tqdm

def train(dataset, model, lr=2e-5, warmup_steps=5000, output_dir="./models/gpt_mlp/checkpoints_nws/", output_prefix="ki"):

    device = torch.device('cuda:0')
    model = model.to(device)
    model.train()

    #Training parameters
    batch_size = 8
    epochs = 10
    optimizer = AdamW(model.parameters(), lr=lr)
    
    #Pre-training processings
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs*len(train_dataloader)
    )

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        sys.stdout.flush()

        progress = tqdm(total=len(train_dataloader), desc=output_prefix)

        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()

            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)

            logits = outputs.logits[:, dataset.prefix_length-1:-1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress.set_postfix({"loss":loss.item()})
            progress.update()

            if (idx + 1) % 1000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        
        progress.close()

        torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model

def main(file_name, model):
    """

    """
    prefix_length = 10

    dataset = ClipPoemDataset(f"data/{file_name}", prefix_length)

    if model == 1:
        model = GPTMLPModel(prefix_length)
    elif model == 2:
        model = GPTTransformerModel()
    else:
        return -1
        
    print("All locked and loaded")
    train(dataset, model)
    

if __name__ == "__main__":
    embeddings = sys.argv[1]
    model = sys.argv[2]
    sys.exit(main(embeddings, int(model)))



