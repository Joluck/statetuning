# RWKV State Tuning 训练脚本

这是一个简单的 RWKV 模型 State Tuning 训练脚本，支持 bf16/fp16/fp32 精度训练。

## 项目结构

```
statetuning/
├── train.py          # 主训练脚本
├── dataset.py        # 数据加载器 (JSONL 格式)
├── model/            # RWKV7 模型定义
│   └── rwkv7/
├── outmodel/         # 训练输出目录 (自动创建)
└── README.md         # 本文件
```

## 快速开始

### 1. 配置训练参数

编辑 `train.py` 中的两个配置类：

```python
# 模型配置
class ModelArgs:
    vocab_size = 65536      # 词表大小 (根据模型调整)
    n_embd = 1024           # 嵌入维度
    n_layer = 24            # 层数
    ctx_len = 512           # 上下文长度
    head_size_a = 64
    head_size_divisor = 8

# 训练配置
class TrainArgs:
    load_model = '/path/to/your/model.pth'  # 预训练模型路径
    data_path = '/path/to/your/data.jsonl'  # 训练数据路径
    output_dir = './outmodel'               # 输出目录
    precision = 'bf16'                      # bf16 | fp16 | fp32
    batch_size = 4
    num_steps = 1000                        # 训练步数
    num_epochs = 1
    learning_rate = 1e-5
```

### 2. 准备数据

数据格式为 JSONL，每行一个样本：

```json
{"text": "User: 你好\n\nAssistant: 你好！有什么可以帮助你的吗？\n\n"}
{"text": "User: 讲个笑话\n\nAssistant: 好的，这是一个笑话...\n\n"}
```

### 3. 运行训练

```bash
cd /path/to/statetuning
python train.py
```

## 配置说明

### ModelArgs (模型配置)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `vocab_size` | 词表大小，需与预训练模型一致 | 65536 |
| `n_embd` | 嵌入维度 | 1024 |
| `n_layer` | Transformer 层数 | 24 |
| `ctx_len` | 最大上下文长度 | 512 |
| `head_size_a` | 注意力头大小 | 64 |
| `head_size_divisor` | LayerNorm eps 除数 | 8 |

### TrainArgs (训练配置)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `load_model` | 预训练模型权重路径 | - |
| `data_path` | 训练数据路径 (JSONL) | - |
| `output_dir` | 输出目录 | ./outmodel |
| `precision` | 训练精度: bf16/fp16/fp32 | bf16 |
| `batch_size` | 批次大小 | 4 |
| `num_steps` | 总训练步数 | 1000 |
| `num_epochs` | 训练轮数 | 1 |
| `learning_rate` | 学习率 | 1e-5 |

## 训练特性

- **State Tuning**: 只训练包含 "state" 的权重，冻结其他所有参数
- **自动混合精度**: 支持 bf16/fp16 训练，节省显存
- **进度条显示**: 使用 tqdm 显示训练进度和损失
- **权重保存**: 自动保存训练后的 state 权重到输出目录

## 输出文件

训练完成后，会在 `output_dir` 目录下生成：

```
outmodel/
└── rwkv7-xxx.state    # 训练后的 state 权重
```

## 注意事项

1. **模型权重匹配**: 确保 `ModelArgs` 中的参数与预训练模型一致
2. **Tokenizer**: 默认使用 `RWKV/rwkv-5-world-3b`，如需更换请修改 `dataset.py`
3. **显存**: 如果显存不足，请减小 `batch_size` 或 `ctx_len`
4. **数据格式**: 确保 JSONL 文件每行是一个有效的 JSON 对象，包含 "text" 字段
