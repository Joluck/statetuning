import json
import random
import torch


def get_tokenizer(tokenizer_path='RWKV/rwkv-5-world-3b'):
    """加载 Hugging Face Tokenizer"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return tokenizer


class JSONLDataset:
    """JSONL 数据加载器"""
    def __init__(self, file_path, tokenizer=None, tokenizer_path='RWKV/rwkv-5-world-3b'):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item['text'])

        # 如果未提供 tokenizer，则从 HF 加载
        if tokenizer is None:
            try:
                self.tokenizer = get_tokenizer(tokenizer_path)
                print(f"Loaded tokenizer from {tokenizer_path}")
            except Exception as e:
                print(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer

        print(f"Loaded {len(self.data)} samples from {file_path}")

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size=4, seq_len=128, vocab_size=65536):
        seq_len+=1
        """获取一个 batch 的数据"""
        # 随机采样
        samples = random.choices(self.data, k=batch_size)

        if self.tokenizer is None:
            # 如果没有 tokenizer，返回随机生成的 token ids（临时方案）
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                labels = labels.cuda()
            return input_ids, labels

        # Tokenize
        input_ids_list = []
        labels_list = []
        for text in samples:
            # HF tokenizer encode returns list by default
            tokens = self.tokenizer.encode(text)
            # Truncate if needed
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            # Padding
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
