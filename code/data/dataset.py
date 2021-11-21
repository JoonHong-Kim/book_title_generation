from torch.utils.data import Dataset, DataLoader, IterableDataset
import os
import torch
import numpy as np

# 참고: https://github.com/seujung/KoBART-translation/blob/main/dataset.py
class Dataset(Dataset):
    """
    data를 model에 넣기위해 사용되는 dataset class
    """

    def __init__(self, dataset, tokenizer):
        super.__init__()
        self.dataset = dataset

        self.tokenizer = tokenizer

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[: self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[: self.max_len]

        return inputs

    def __getitem__(self, index):
        item = self.dataset.iloc[index]
        input_ids = self.tokenizer.encode(item["descriptions"], add_special_tokens=True)
        input_ids = self.add_padding_data(input_ids)

        labels = self.tokenizer.encode(item["title"], add_special_tokens=True)

        decoder_input_ids = [self.tokenizer.pad_token_id]
        decoder_input_ids += labels[:-1]
        decoder_input_ids = self.add_padding_data(decoder_input_ids)

        labels = self.add_ignored_data(labels)

        return {
            "input_ids": np.array(input_ids, dtype=np.int_),
            "decoder_input_ids": np.array(decoder_input_ids, dtype=np.int_),
            "labels": np.array(labels, dtype=np.int_),
        }

    def __len__(self):
        return len(self.dataset)
