from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.optim import Adam
import wandb
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import os
from set_seed import set_seed

from models import SBERT_with_KLUE_BERT, SBERT_with_ROBERTA_LARGE, SBERT_with_KOELECTRA_BASE
from datasets import KorSTSDatasets, KorSTS_collate_fn, bucket_pair_indices

models = {"klue/bert-base": SBERT_with_KLUE_BERT, "klue/roberta-large": SBERT_with_ROBERTA_LARGE, 
        "monologg/koelectra-base-discriminator": SBERT_with_KOELECTRA_BASE,
        "monologg/koelectra-base-v2-discriminator": SBERT_with_KOELECTRA_BASE,
        "monologg/koelectra-base-v3-discriminator": SBERT_with_KOELECTRA_BASE,
        }


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print("training on", device)

    if not config["test_mode"]:
        run = wandb.init(project="sentence_bert", entity="intrandom5", config=config, name=config['log_name'])

    train_datasets = KorSTSDatasets(config['train_x_dir'], config['train_y_dir'])
    valid_datasets = KorSTSDatasets(config['valid_x_dir'], config['valid_y_dir'])
    print(f"train_x dataset: {config['train_x_dir']}")
    print(f"train_y dataset: {config['train_y_dir']}")
    print(f"valid_x dataset: {config['valid_x_dir']}")
    print(f"valid_y dataset: {config['valid_y_dir']}")

    # train_seq_lengths = [(len(s1), len(s2)) for (s1, s2) in train_datasets.x]
    # train_sampler = bucket_pair_indices(train_seq_lengths, batch_size=config['batch_size'], max_pad_len=10)

    train_loader = DataLoader(
        train_datasets, 
        collate_fn=KorSTS_collate_fn, 
        shuffle=True,
        batch_size=config['batch_size'],
        # batch_sampler=train_sampler
    )
    valid_loader = DataLoader(
        valid_datasets,
        collate_fn=KorSTS_collate_fn,
        batch_size=config['batch_size']
    )

    if config['base_model'].startswith("monologg/koelectra"):
        model = models[config['base_model']](version=config["base_model"].split("-")[2])
    else:
        model = models[config['base_model']]()
        
    print("Base model is", config['base_model'])
    if os.path.exists(config["model_load_path"]):
        model.load_state_dict(torch.load(config["model_load_path"]))
        print("weights loaded from", config["model_load_path"])
    else:
        print("no pretrained weights provided.")
    model.to(device)

    epochs = config['epochs']
    criterion = nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=config['lr'])

    pbar = tqdm(range(epochs))

    for epoch in pbar:
        for iter, data in enumerate(tqdm(train_loader)):
            s1, s2, label = data
            s1 = s1.to(device)
            s2 = s2.to(device)
            label = label.to(device)
            
            logits = model(s1, s2)
            loss = criterion(logits.squeeze(-1), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            if not config["test_mode"]:
                wandb.log({"train_loss": loss})
            pbar.set_postfix({"train_loss": loss})

        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(valid_loader)):
                s1, s2, label = data
                s1 = s2.to(device)
                s2 = s2.to(device)
                label = label.to(device)
                
                logits = model(s1, s2)
                loss = criterion(logits.squeeze(-1), label)
                val_loss += loss.detach().item()
        val_loss = val_loss/i
        if not config["test_mode"]:
            wandb.log({"valid_loss": val_loss, "epoch": epoch})
        pbar.set_postfix({"valid_loss": val_loss, "epoch": epoch})
        
    torch.save(model.state_dict(), config["model_save_path"])


if __name__ == "__main__":
    # 결과 재현성을 위한 랜덤 시드 고정.
    set_seed(13)

    parser = argparse.ArgumentParser(description='Training SBERT.')
    parser.add_argument("--conf", type=str, default="sbert_config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
    