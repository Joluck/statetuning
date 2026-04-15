import os
os.environ["WKV"] = "CUDA"  # 必须在导入模型前设置
os.environ["FUSED_KERNEL"] = "0"
os.environ["RWKV_TRAIN_TYPE"] = "none"
os.environ["RWKV_HEAD_SIZE_A"]='64'
os.environ["RWKV_MY_TESTING"]='x070'
os.environ["RWKV_TRAIN_TYPE"]='state'
os.environ["RWKV_FLOAT_MODE"] = 'bf16'
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from model.rwkv7.model import RWKV7
from dataset import JSONLDataset
from tqdm import tqdm
import os
import json

parser = argparse.ArgumentParser()
# 模型参数
parser.add_argument('--vocab_size', type=int, default=65536, help='词表大小')
parser.add_argument('--n_embd', type=int, default=1024, help='嵌入维度')
parser.add_argument('--n_layer', type=int, default=24, help='层数')
# 训练参数
parser.add_argument('--load_model', type=str, default='/home/rwkv/models/rwkv7/rwkv7-g1a-0.4b-20250905-ctx4096.pth', help='预训练模型路径')
parser.add_argument('--data_path', type=str, default='/home/rwkv/datas/bad_lan.jsonl', help='训练数据路径')
parser.add_argument('--output_dir', type=str, default='./outmodel', help='输出文件夹路径')
parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'], help='训练精度')
parser.add_argument('--batch_size', type=int, default=4, help='batch大小')
parser.add_argument('--num_epochs', type=int, default=1, help='训练轮数')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
parser.add_argument('--ctx_len', type=int, default=2048, help='上下文长度')
parser.add_argument('--shuffle', action='store_true', help='是否打乱数据顺序')
args = parser.parse_args()

# 模型配置类
class ModelArgs:
    vocab_size = args.vocab_size
    n_embd = args.n_embd
    n_layer = args.n_layer
    head_size_a = 64
    head_size_divisor = 8

# 训练配置类
class TrainArgs:
    load_model = args.load_model
    data_path = args.data_path
    output_dir = args.output_dir
    precision = args.precision
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    ctx_len = args.ctx_len
    shuffle = args.shuffle
# 创建配置
model_args = ModelArgs()
model_args.dim_att = model_args.n_embd
model_args.dim_ffn = model_args.n_embd * 4
train_args = TrainArgs()

# 创建模型
model = RWKV7(model_args)
model = model.cuda() if torch.cuda.is_available() else model
state_dict = torch.load(train_args.load_model, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)

# 根据 precision 参数设置训练精度
if train_args.precision == 'bf16':
    model = model.to(torch.bfloat16)
elif train_args.precision == 'fp16':
    model = model.to(torch.float16)
# fp32 不需要转换，默认就是 float32

# 冻结除包含 "state" 以外的所有权重
for name, param in model.named_parameters():
    if 'state' not in name.lower():
        param.requires_grad = False

# 查看模型权重组件及梯度状态
print("\n" + "="*60)
print("模型权重组件及梯度状态:")
total_params = 0
trainable_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
print("="*60)
print(f"total params: {total_params:,}")
print(f"Frozen_params: {total_params - trainable_params:,}")
print(f"Train_params: {trainable_params:,}")
print("="*60 + "\n")

# 创建优化器
optimizer = Adam(model.parameters(), lr=train_args.learning_rate)

# 加载数据集
dataset = JSONLDataset(train_args.data_path, shuffle=train_args.shuffle)

print("\n" + "="*50)
print("数据集样本示例:")
for i in range(min(2, len(dataset))):
    print(f"\n样本 {i+1}:")
    print(dataset.data[i][:200] + "..." if len(dataset.data[i]) > 200 else dataset.data[i])
print("="*50 + "\n")

seq_len = train_args.ctx_len
def get_batch(batch_size=train_args.batch_size, seq_len=seq_len):
    return dataset.get_batch(batch_size, seq_len, model_args.vocab_size)

# 训练循环
model.train()
os.makedirs(train_args.output_dir, exist_ok=True)
loss_log_path = os.path.join(train_args.output_dir, 'train_loss.jsonl')

steps_per_epoch = max(1, len(dataset) // train_args.batch_size)
total_steps = steps_per_epoch * train_args.num_epochs
global_step = 0

model_base_name = os.path.basename(train_args.load_model).replace('.pth', '')

def save_state_weights(suffix=''):
    state_dict_to_save = {}
    for name, param in model.named_parameters():
        if 'state' in name.lower():
            clean_name = name.replace("model.", "")
            state_dict_to_save[clean_name] = param.cpu()
    save_name = f"{model_base_name}{suffix}.pth"
    save_path = os.path.join(train_args.output_dir, save_name)
    torch.save(state_dict_to_save, save_path)
    print(f"已保存 {len(state_dict_to_save)} 个 state 权重到: {save_path}")

with open(loss_log_path, 'w', encoding='utf-8') as log_f:
    for epoch in range(train_args.num_epochs):
        dataset._reset_indices()
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{train_args.num_epochs}", ncols=100)
        for step in pbar:
            input_ids, labels = get_batch()

            logits = model(input_ids)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            log_f.write(json.dumps({"epoch": epoch, "step": global_step, "loss": loss_val}) + '\n')
            log_f.flush()
            global_step += 1

        save_state_weights(f"-{epoch+1}")

print(f"Training completed! Total steps: {global_step}")
