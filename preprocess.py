from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import argparse
import yaml


def prepare_data(tsv_dict):
    label = tsv_dict["label"]
    sentence1 = tsv_dict["sentence_1"]
    sentence2 = tsv_dict["sentence_2"]
    x = [(tokenizer.encode(s1), tokenizer.encode(s2)) for s1, s2 in zip(sentence1, sentence2)]
    assert len(x) == len(label)
    return x, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training SBERT.')
    parser.add_argument("--conf", type=str, default="sbert_config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    train_tsv = pd.read_csv(config["train_csv"])
    valid_tsv = pd.read_csv(config["valid_csv"])

    train_x, train_y = prepare_data(train_tsv)
    valid_x, valid_y = prepare_data(valid_tsv)

    np.save(config["train_x_dir"], np.array(train_x, dtype='object'))
    np.save(config["train_y_dir"], np.array(train_y, dtype='object'))

    np.save(config["valid_x_dir"], np.array(valid_x, dtype='object'))
    np.save(config["valid_y_dir"], np.array(valid_y, dtype='object'))
