from torch.utils.data import Dataset
import numpy as np

# 참고: https://github.com/seujung/KoBART-translation/blob/main/dataset.py
class KoBARTDataset(Dataset):
    """
    data를 model에 넣기위해 사용되는 dataset class
    """

    def __init__(self, dataset, tokenizer):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer

    def add_padding_data(self, inputs):
        if len(inputs) < 512:
            pad = np.array([self.tokenizer.pad_token_id] * (512 - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:512]

        return inputs

    def __getitem__(self, index):
        item = self.dataset.iloc[index]
        input_ids = self.tokenizer.encode(item["description"], add_special_tokens=True)
        input_ids = self.add_padding_data(input_ids)

        labels = self.tokenizer.encode(item["title"], add_special_tokens=True)
        labels = self.add_padding_data(labels)
        decoder_input_ids = [self.tokenizer.pad_token_id]
        decoder_input_ids += labels[:-1]

        return {
            "input_ids": np.array(input_ids, dtype=np.int_),
            "decoder_input_ids": np.array(decoder_input_ids, dtype=np.int_),
            "labels": np.array(labels, dtype=np.int_),
        }

    def __len__(self):
        return len(self.dataset)
