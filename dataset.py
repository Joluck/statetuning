import json
import random
import torch


def get_tokenizer(tokenizer_path='rwkv_vocab_v20230424'):
    from tokenizer.utils import PIPELINE
    return PIPELINE(model=None, WORD_NAME=tokenizer_path)


class JSONLDataset:
    """JSONL 数据加载器"""
    def __init__(self, file_path, tokenizer=None, tokenizer_path='rwkv_vocab_v20230424', shuffle=True):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item['text'])

        if tokenizer is None:
            try:
                self.tokenizer = get_tokenizer(tokenizer_path)
                print(f"Loaded tokenizer from {tokenizer_path}")
            except Exception as e:
                raise ValueError(f"Failed to load tokenizer: {e}")
        else:
            self.tokenizer = tokenizer

        self.shuffle = shuffle
        self._indices = []
        self._pos = 0
        self._reset_indices()

        print(f"Loaded {len(self.data)} samples from {file_path} (shuffle={shuffle})")

    def _reset_indices(self):
        self._indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self._indices)
        self._pos = 0

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size=4, seq_len=128, vocab_size=65536):
        seq_len += 1

        if self.tokenizer is None:
            raise ValueError("tokenizer is None, please provide a valid tokenizer")

        # 取 batch_size 个样本，遍历完则重置
        if self._pos + batch_size > len(self._indices):
            self._reset_indices()
        batch_indices = self._indices[self._pos:self._pos + batch_size]
        self._pos += batch_size
        samples = [self.data[i] for i in batch_indices]

        input_ids_list = []
        labels_list = []
        for text in samples:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            if len(tokens) < seq_len:
                tokens = tokens + [0] * (seq_len - len(tokens))
            input_ids_list.append(tokens[:-1])
            labels_list.append(tokens[1:])

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        return input_ids, labels
