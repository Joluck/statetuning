import os
os.environ["WKV"] = "CUDA"  # 必须在导入模型前设置
os.environ["FUSED_KERNEL"] = "0"
os.environ["RWKV_TRAIN_TYPE"] = "none"
os.environ["RWKV_HEAD_SIZE_A"]='64'
os.environ["RWKV_MY_TESTING"]='x070'
os.environ["RWKV_TRAIN_TYPE"]='state'
os.environ["RWKV_FLOAT_MODE"] = 'bf16'
import torch
import torch.nn as nn
from torch.optim import Adam
from model.rwkv7.model import RWKV7
from dataset import JSONLDataset
from tqdm import tqdm
import os
# 模型配置类
class ModelArgs:
    vocab_size = 65536      # 词表大小
    n_embd = 1024           # 嵌入维度
    n_layer = 24            # 层数
    ctx_len = 512           # 上下文长度
    head_size_a = 64        # 头大小
    head_size_divisor = 8   # 头大小除数 (用于LayerNorm eps)

# 训练配置类
class TrainArgs:
    load_model = '/home/rwkv/models/rwkv7/rwkv7-g1a-0.4b-20250905-ctx4096.pth'  # 预训练模型路径
    data_path = '/home/rwkv/datas/bad_lan.jsonl'
    output_dir = './outmodel'  # 输出文件夹路径，自动创建
    precision = 'bf16'        # 训练精度: 'bf16' | 'fp16' | 'fp32'
    batch_size = 4            # batch大小
    num_steps = 1000          # 训练步数
    num_epochs = 1            # 训练轮数
    learning_rate = 1e-5      # 学习率
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
dataset = JSONLDataset(train_args.data_path)

print("\n" + "="*50)
print("数据集样本示例:")
for i in range(min(2, len(dataset))):
    print(f"\n样本 {i+1}:")
    print(dataset.data[i][:200] + "..." if len(dataset.data[i]) > 200 else dataset.data[i])
print("="*50 + "\n")

seq_len = model_args.ctx_len
def get_batch(batch_size=train_args.batch_size, seq_len=seq_len):
    return dataset.get_batch(batch_size, seq_len, model_args.vocab_size)

# 训练循环
model.train()

# 使用 tqdm 进度条
pbar = tqdm(range(train_args.num_steps), desc="Training", ncols=100)
for step in pbar:
    # 获取数据
    input_ids, labels = get_batch()

    # 前向传播
    logits = model(input_ids)

    # 计算损失 (交叉熵)
    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 更新进度条显示
    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

print("Training completed!")

# 保存包含 "state" 的权重
print("\n保存包含 'state' 的权重...")
os.makedirs(train_args.output_dir, exist_ok=True)
model_name = os.path.basename(train_args.load_model).replace('.pth', '.state')
save_path = os.path.join(train_args.output_dir, model_name)
state_dict_to_save = {}
for name, param in model.named_parameters():
    if 'state' in name.lower():
        # 去掉 "model." 前缀
        clean_name = name.replace("model.", "")
        state_dict_to_save[clean_name] = param.cpu()

torch.save(state_dict_to_save, save_path)
print(f"已保存 {len(state_dict_to_save)} 个 state 权重到: {save_path}")
